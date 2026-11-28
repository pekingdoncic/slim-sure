"""Keep exact weights for every scored full baseline checkpoint for later constrained selection."""
import fcntl
import hashlib
import os
import time

from workflow import EXP, REPORT, STATE, now, read, run_dir, save


def preserve(run, step, identity):
    folder = run / "validation" / f"update_{step:08d}"
    destination = folder / "selection_weights.pt"
    receipt = folder / "selection_weights.json"
    if destination.exists():
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if receipt.exists():
            record = read(receipt)
            if (record["sha256"], record["identity"], record["step"]) != (digest, identity, step):
                raise RuntimeError(f"Validation weight identity mismatch at {step}")
        else:
            import torch
            item = torch.load(destination, map_location="cpu")
            if (item["identity"], item["step"]) != (identity, step):
                raise RuntimeError(f"Uncommitted validation archive mismatch at {step}")
            save(receipt, dict(step=step, identity=identity, sha256=digest, saved_utc=now()))
        return True
    sources = [run / "checkpoint_last.pt", run / "checkpoint_best_general.pt",
               run / "checkpoint_best_lowtex.pt"]
    for source in sources:
        if not source.is_file():
            continue
        import torch
        with source.open("rb") as stream:
            saved = torch.load(stream, map_location="cpu")
        if saved["step"] != step or saved["identity"] != identity:
            continue
        folder.mkdir(exist_ok=True, parents=True)
        tmp = destination.with_suffix(".tmp")
        torch.save(dict(identity=identity, step=step, model=saved["model"]), tmp)
        tmp.replace(destination)
        save(folder / "selection_weights.json", dict(step=step, identity=identity,
             sha256=hashlib.sha256(destination.read_bytes()).hexdigest(), saved_utc=now()))
        return True
    return False


def main():
    with (REPORT / "development_archive.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        budget = read(EXP / "manifests/full_budget.json")
        marks = sorted({*range(0, budget["optimizer_updates"] + 1, budget["validation_every"]),
                        budget["optimizer_updates"]})
        index = 0
        while index < len(marks):
            state = read(STATE)
            run = run_dir()
            if not run or not (run / "progress.json").exists():
                if state["current_phase"] != "02":
                    return
                time.sleep(20)
                continue
            step = marks[index]
            committed = read(run / "progress.json")["step"]
            identity = read(REPORT / "full_run.json")["identity"]
            if committed >= step:
                if preserve(run, step, identity):
                    save(REPORT / "development_archive.json", dict(phase="02", latest=step,
                         run=str(run.relative_to(EXP)), pid=os.getpid(), captured_utc=now()))
                    index += 1
                    continue
                if committed > step:
                    raise RuntimeError(f"Missing archived full validation weights at {step}; committed {committed}")
            time.sleep(20)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        save(REPORT / "development_archive_error.json", dict(error=repr(error), utc=now()))
        raise
