"""Bounded current-boot storage/resource monitor; never changes host/WSL configuration."""
import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

FAULT = re.compile(r"I/O error|Buffer I/O|blk_update_request|device.*offline|offlined|SIGBUS|Bus error|swap.*(?:error|fail)|storvsc.*(?:fail|error)|EXT4-fs.*error|I/O.*timed? out|Remounting filesystem read-only", re.I)
NEW_WARNING = re.compile(r"journal.*corrupt|corrupt.*journal|uncleanly shut down", re.I)
MAX_KERNEL_RECORDS = 2000


def system_sample():
    mem = {k: int(v.strip().split()[0])*1024 for k,v in
           (line.split(":",1) for line in Path("/proc/meminfo").read_text().splitlines())}
    vm = dict(line.split() for line in Path("/proc/vmstat").read_text().splitlines())
    host = shutil.disk_usage("/mnt/c").free if Path("/mnt/c").is_mount() else None
    return dict(mem_total_bytes=mem["MemTotal"], mem_available_bytes=mem["MemAvailable"],
                swap_total_bytes=mem["SwapTotal"], swap_free_bytes=mem["SwapFree"],
                swap_used_bytes=mem["SwapTotal"]-mem["SwapFree"],
                pswpin=int(vm["pswpin"]), pswpout=int(vm["pswpout"]),
                host_c_free_bytes=host, monotonic=time.monotonic())


def _save(path,value):
    tmp=path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(value,f,indent=2,sort_keys=True);f.write("\n");f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


class EnvironmentMonitor:
    def __init__(self,run,phase,minimum_host_bytes=2*2**30):
        self.run=Path(run);self.out=self.run/"environment_probe"/phase
        self.out.mkdir(parents=True,exist_ok=True)
        self.minimum_host_bytes=minimum_host_bytes
        self.boot=None;self.cursor=None;self.started=time.monotonic();self.checks=0
        self.last_sample=None;self.blocked_reason=None;self.status=None
        self.check(initial=True)

    def _persist(self,reason,faults,details=None):
        status=dict(status="blocked" if reason else "pass",scope="bounded current-boot checks only",
                    boot_id=self.boot,checks=self.checks,latest=self.last_sample,new_faults=faults,
                    reason=reason,elapsed_seconds=time.monotonic()-self.started,
                    journal_cursor=self.cursor,maximum_kernel_records_per_check=MAX_KERNEL_RECORDS,
                    storage_root_cause_resolved=False)
        if details:status["details"]=details
        self.status=status
        if reason:
            self.blocked_reason=reason
            # Preserve every blocking event even if a later invocation reuses this phase directory.
            _save(self.out/("incident_"+str(time.time_ns())+".json"),status)
        _save(self.out/"status.json",status)
        return status

    def record_failure(self,reason,details=None):
        return self._persist(reason,[],details)

    def check(self,initial=False):
        if self.blocked_reason:
            raise RuntimeError("Environment guard remains blocked: "+self.blocked_reason)
        self.checks+=1
        faults=[]
        try:
            current_boot=Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            if not current_boot:raise RuntimeError("Cannot determine WSL boot identity")
            if self.boot is None:self.boot=current_boot
            elif current_boot!=self.boot:raise RuntimeError("WSL boot changed during execution")
            command=["journalctl","-k","-b","--no-pager","-n",str(MAX_KERNEL_RECORDS+1),"-o","json"]
            if self.cursor:command.append("--after-cursor="+self.cursor)
            proc=subprocess.run(command,capture_output=True,text=True,timeout=10)
            (self.out/("kernel_before.log" if initial else "kernel_latest.log")).write_text(proc.stdout)
            if proc.returncode:raise RuntimeError("Cannot verify current-boot kernel log: "+proc.stderr.strip())
            lines=[line for line in proc.stdout.splitlines() if line.strip()]
            if len(lines)>MAX_KERNEL_RECORDS:
                raise RuntimeError("Kernel journal exceeds bounded review limit; continuity cannot be established")
            records=[json.loads(line) for line in lines]
            if initial and not records:raise RuntimeError("Current-boot kernel journal is empty")
            for record in records:
                if record.get("_BOOT_ID")!=self.boot.replace("-",""):
                    raise RuntimeError("Kernel record has a different or missing boot identity")
                if not record.get("__CURSOR"):raise RuntimeError("Kernel record lacks a journal cursor")
                message=record.get("MESSAGE")
                if not isinstance(message,str):raise RuntimeError("Cannot decode kernel log message")
                if FAULT.search(message) or (not initial and NEW_WARNING.search(message)):
                    faults.append(dict(cursor=record["__CURSOR"],message=message,
                                       monotonic_us=record.get("__MONOTONIC_TIMESTAMP"),
                                       realtime_us=record.get("__REALTIME_TIMESTAMP")))
            if records:self.cursor=records[-1]["__CURSOR"]
            sample=system_sample()
            sample.update(boot_id=self.boot,ext4_free_bytes=shutil.disk_usage(self.run).free,
                          utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                          kernel_records_since_previous_check=len(records))
            self.last_sample=sample
            with (self.out/"resources.jsonl").open("a") as f:
                f.write(json.dumps(sample,sort_keys=True)+"\n");f.flush();os.fsync(f.fileno())
            reason=None
            if faults:reason="Storage fault in current boot" if initial else "New current-boot storage fault"
            elif sample["host_c_free_bytes"] is None:reason="Cannot verify WSL backing C: volume free space"
            elif sample["host_c_free_bytes"]<self.minimum_host_bytes:reason="Host C: free space below required reserve"
            elif sample["ext4_free_bytes"]<self.minimum_host_bytes:reason="Guest filesystem free space below required reserve"
            elif sample["mem_available_bytes"]<512*2**20:reason="Available memory below bounded execution reserve"
            status=self._persist(reason,faults,dict(initial_snapshot=initial,journal_format="json",
                initial_unclean_journal_messages=[r["MESSAGE"] for r in records if initial and NEW_WARNING.search(r["MESSAGE"])]))
            if reason:raise RuntimeError(reason)
            return status
        except BaseException as exc:
            if not self.blocked_reason:
                self._persist("Environment monitoring failed: "+str(exc),faults,
                              dict(exception_type=type(exc).__name__,initial_snapshot=initial))
            raise
