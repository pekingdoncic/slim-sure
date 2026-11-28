import os
import subprocess
import threading
import time
import torch

GIB = 1024**3


def smi(query,fields):
    result = subprocess.run(["nvidia-smi","--query-"+query+"="+fields,"--format=csv,noheader,nounits"],
                            text=True,capture_output=True,check=True,timeout=12)
    return [[s.strip() for s in line.split(",")] for line in result.stdout.splitlines() if line.strip()]


class MemoryPressure(RuntimeError):
    pass


class Guard:
    """Allocator cap with independent driver polling, device free space and process accounting."""
    def __init__(self,requested):
        if not 0<requested<=30 or not torch.cuda.is_available():
            raise RuntimeError("CUDA and a positive requested cap of at most 30 GiB are required")
        if torch.cuda.device_count()!=1:
            raise ValueError("select exactly one visible GPU with --gpu")
        free,total = torch.cuda.mem_get_info(0)
        self.cap = min(requested,free/GIB-4.)
        self.target = self.cap-2.
        if self.target<=0:
            raise MemoryPressure(f"insufficient free memory ({free/GIB:.2f} GiB)")
        torch.cuda.set_per_process_memory_fraction(self.cap/(total/GIB),device=0)
        devices = smi("gpu","index,uuid,name,memory.total,memory.free,utilization.gpu")
        ident = str(getattr(torch.cuda.get_device_properties(0),"uuid","")).lower().removeprefix("gpu-")
        selected = os.environ.get("CUDA_VISIBLE_DEVICES","0")
        rows = [r for r in devices if r[1].lower().removeprefix("gpu-")==ident]
        if not rows:
            rows = [r for r in devices if r[0]==selected or r[1].startswith(selected)]
        if len(rows)!=1:
            raise RuntimeError(f"cannot identify selected CUDA device {selected}")
        self.uuid,self.name = rows[0][1:3]
        self.total,self.initial_free = total/GIB,free/GIB
        self.last,self.error = {},None
        self.peak_process,self.util_sum,self.samples = None,0.,0
        self.mutex = threading.Lock()
        self.stop = threading.Event()
        self.poll()
        self.thread = threading.Thread(target=self.watch,daemon=True)
        self.thread.start()

    def poll(self):
        row = next(r for r in smi("gpu","uuid,memory.free,utilization.gpu") if r[0]==self.uuid)
        processes = smi("compute-apps","gpu_uuid,pid,used_gpu_memory")
        own = [float(r[2])/1024 for r in processes if r[0]==self.uuid and r[1]==str(os.getpid())]
        process = max(own,default=None)
        with self.mutex:
            self.last = dict(uuid=self.uuid,name=self.name,total_gib=self.total,initial_free_gib=self.initial_free,
                             cap_gib=self.cap,target_gib=self.target,free_gib=float(row[1])/1024,
                             utilization=float(row[2]),process_gib=process,pid=os.getpid())
            if process is not None:
                self.peak_process = max(self.peak_process or 0.,process)
            self.util_sum += self.last["utilization"]
            self.samples += 1

    def watch(self):
        while not self.stop.wait(1.):
            try:
                self.poll()
            except Exception as e:
                self.error = str(e)
                return

    def check(self):
        free = torch.cuda.mem_get_info(0)[0]/GIB
        if self.error:
            raise MemoryPressure("GPU telemetry failed: "+self.error)
        own=self.last["process_gib"]
        # Driver telemetry uses a simultaneous free/own snapshot; own allocations cancel.
        shared_shortfall=(own is not None and self.last["free_gib"]+own-4 < self.cap-.25)
        if free<4 or (own is not None and own>self.cap) or shared_shortfall:
            raise MemoryPressure(f"shared GPU headroom changed: free={free:.2f}, telemetry={self.last}")

    def reset_observations(self):
        with self.mutex:
            self.peak_process=self.last.get("process_gib")
            self.util_sum,self.samples=0.,0

    def snapshot(self):
        with self.mutex:
            return dict(**self.last,peak_process_gib=self.peak_process,utilization_mean=self.util_sum/max(1,self.samples),
                        allocated_gib=torch.cuda.max_memory_allocated()/GIB,reserved_gib=torch.cuda.max_memory_reserved()/GIB)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=15)


def precision_name(request):
    if request in ("bf16","bfloat16"):
        return "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    if request in ("fp16","float16"):
        return "float16"
    raise ValueError(request)
