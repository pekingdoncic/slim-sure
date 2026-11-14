"""Identity, atomic checkpoints, frozen backbone and numerical policy."""
import ast
import contextlib
import datetime
import fcntl
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import resource
import sys
import time
import uuid
import numpy as np
import torch

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    with tmp.open("w") as f:
        json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def append_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(obj,sort_keys=True,allow_nan=False)+"\n");f.flush()

def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()

@contextlib.contextmanager
def run_lock(run):
    path=Path(run)/"writer.lock";path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a+") as f:
        try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError("Another process owns this run")
        f.seek(0);f.truncate();f.write(str(os.getpid()));f.flush()
        try: yield
        finally: fcntl.flock(f,fcntl.LOCK_UN)

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())

def set_rng(state):
    random.setstate(state["python"]);np.random.set_state(state["numpy"]);torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])

def state_hash(module):
    h=hashlib.sha256()
    for name,tensor in sorted(module.state_dict().items()):
        h.update(name.encode());h.update(str(tensor.dtype).encode());h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()

def code_identity(root):
    root=Path(root)
    paths=[root/"reliability_stage.py",root/"reliability_data.py"]
    for folder in ("src","configs","reliability_v1","utils"):
        paths += [p for p in (root/folder).rglob("*.py") if "__pycache__" not in p.parts]
    return {str(p.relative_to(root)):sha(p) for p in sorted(set(paths)) if p.is_file()}

def identity(config,data_identity):
    versions={}
    for name in ("torch","numpy","mamba-ssm","triton","h5py","yacs","opencv-python-headless"):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    kernels={}
    distribution=importlib.metadata.distribution("mamba-ssm")
    for symbol in json.loads(Path(config["profile"]).read_text())["autotuners"]:
        module=symbol.rsplit(".",1)[0]
        path=distribution.locate_file(module.replace(".","/")+".py")
        kernels[module]=sha(path)
    return dict(environment=dict(executable=sys.executable,versions=versions,cuda_build=torch.version.cuda,kernel_source_sha256=kernels),schema="reliability_v1_identity_1",code=code_identity(config["repo_root"]),
        config_sha=digest({k:v for k,v in config.items() if k not in ("run_stage","run_dir")}),
        data=data_identity,original_weight_sha=sha(config["checkpoint"]),profile_sha=sha(config["profile"]),
        numerical={k:config[k] for k in ("amp","tf32","cudnn_benchmark","cudnn_deterministic","deterministic_algorithms","resolution")},
        label_schema="coarse_parent_correctness_v1")

def save_checkpoint(path,head,optimizer,position,run_identity,extra=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    obj=dict(schema="slim_reliability_checkpoint_v1",identity=run_identity,
        head=head.state_dict(),optimizer=optimizer.state_dict(),rng=rng_state(),position=position,extra=extra or {})
    # Content addressed payload; a single atomic pointer commits payload+checksum together.
    tmp=path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    with tmp.open("wb") as f:
        torch.save(obj,f);f.flush();os.fsync(f.fileno())
    checksum=sha(tmp);payload=path.with_name(path.stem+"."+checksum+".pt")
    os.replace(tmp,payload)
    prior=json.loads(path.read_text()).get("payload") if path.exists() else None
    atomic_json(path,dict(schema="atomic_checkpoint_pointer_v1",payload=payload.name,sha256=checksum,identity_sha=digest(run_identity),time=now()))
    if prior and prior!=payload.name:
        old=path.parent/prior
        # Only remove this pointer's superseded content-addressed recovery payload.
        if old.parent.resolve()==path.parent.resolve() and old.name.startswith(path.stem+".") and old.suffix==".pt": old.unlink(missing_ok=True)
    return checksum

def load_checkpoint(path,head,optimizer,run_identity,restore_rng=True):
    path=Path(path);meta=json.loads(path.read_text());payload=path.parent/meta["payload"]
    if payload.parent.resolve()!=path.parent.resolve() or sha(payload)!=meta["sha256"]: raise RuntimeError("Checkpoint checksum/path mismatch")
    if meta["identity_sha"]!=digest(run_identity): raise RuntimeError("Checkpoint identity mismatch")
    obj=torch.load(payload,map_location="cpu",weights_only=False)
    if obj["identity"]!=run_identity: raise RuntimeError("Checkpoint embedded identity mismatch")
    head.load_state_dict(obj["head"],strict=True)
    if optimizer is not None: optimizer.load_state_dict(obj["optimizer"])
    if restore_rng:set_rng(obj["rng"])
    return obj

def resource_sample():
    return dict(time=now(),allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),
        peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
        rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        rss_bytes=int(Path("/proc/self/statm").read_text().split()[1])*os.sysconf("SC_PAGE_SIZE"))

def apply_profile(path):
    profile=json.loads(Path(path).read_text())
    if profile.get("policy")!="fixed_diagnostic_profile_0" or not profile.get("user_approved"):raise RuntimeError("Unapproved profile")
    installed={}
    for symbol,entry in profile["autotuners"].items():
        module_name,name=symbol.rsplit(".",1)
        if not module_name.startswith("mamba_ssm.ops.triton."): raise RuntimeError("Unexpected profile module")
        module=importlib.import_module(module_name);tuner=getattr(module,name)
        values=set(entry["cache"].values());choices=[c for c in tuner.configs if str(c) in values]
        if len(choices)!=1 or len(values)!=1: raise RuntimeError("Profile does not resolve exact installed config")
        tuner.configs=[choices[0]];tuner.cache.clear()
        for key in entry["cache"]:tuner.cache[ast.literal_eval(key)]=choices[0]
        installed[symbol]=dict(configuration=str(choices[0]),source=module.__file__,sha256=sha(module.__file__))
    return dict(source=str(path),sha256=sha(path),kernels=installed,
        scope="Existing profile reused; R832 separately gated by engineering repeats")

def prepare(config,run):
    sys.dont_write_bytecode=True
    for variable,folder in (("TRITON_CACHE_DIR","triton"),("CUDA_CACHE_PATH","cuda"),("TORCH_EXTENSIONS_DIR","extensions"),("XDG_CACHE_HOME","xdg")):
        p=Path(run)/"cache"/folder;p.mkdir(parents=True,exist_ok=True);os.environ[variable]=str(p)
    torch.set_num_threads(1)
    seed_all(config["seed"])
    torch.backends.cuda.matmul.allow_tf32=config["tf32"];torch.backends.cudnn.allow_tf32=config["tf32"]
    torch.backends.cudnn.benchmark=config["cudnn_benchmark"];torch.backends.cudnn.deterministic=config["cudnn_deterministic"]
    torch.use_deterministic_algorithms(config["deterministic_algorithms"])
    if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable; no automatic CPU/precision fallback")
    versions={}
    for name in ("torch","numpy","mamba-ssm","triton","h5py","yacs","opencv-python-headless"):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    installed_packages=sorted([dict(name=d.metadata.get("Name","unknown"),version=d.version) for d in importlib.metadata.distributions()],key=lambda x:x["name"].lower())
    conda_packages=[]
    for path in (Path(sys.prefix)/"conda-meta").glob("*.json"):
        entry=json.loads(path.read_text());conda_packages.append({k:entry.get(k) for k in ("name","version","build")})
    import cv2
    versions["cv2_runtime"]=cv2.__version__
    atomic_json(Path(run)/"environment.json",dict(time=now(),python=sys.version,executable=sys.executable,
        packages=versions,installed_packages=installed_packages,conda_packages=conda_packages,cuda_build=torch.version.cuda,gpu=torch.cuda.get_device_name(),numerical=config,
        conda_environment=os.environ.get("CONDA_DEFAULT_ENV")))
    installed=apply_profile(config["profile"])
    atomic_json(Path(run)/"numerical_profile.json",installed)
    return installed

class Runtime:
    def __init__(self,config,run):
        self.config=config;self.run=Path(run);self.calls=dict(backbone=0,head_forward=0,head_backward=0,original_fine=0,teacher=0,sure_backbone=0)
        started=time.perf_counter()
        prepare(config,run)
        from default_config import get_config
        from src.slim import SLiM
        from src.reliability import build_reliability_head
        native=get_config(config["native_config"]);self.native=native
        model=SLiM(native.MODEL)
        checkpoint=torch.load(config["checkpoint"],map_location="cpu",weights_only=False)
        result=model.load_state_dict(checkpoint["state_dict"],strict=True);del checkpoint
        model=model.cuda().eval().requires_grad_(False)
        model.capture_reliability=True
        model.sparse_direct=config["sparse_direct"]
        self.model=model
        seed_all(config["seed"])
        self.head=build_reliability_head(model.d_coarse,config["reliability"]["architecture"]).cuda().float()
        self.optimizer=torch.optim.AdamW(self.head.parameters(),lr=config["lr"],weight_decay=config["weight_decay"])
        self.frozen_hash=state_hash(model)
        assert all(not p.requires_grad for p in model.parameters())
        assert {id(p) for p in self.head.parameters()}=={id(p) for g in self.optimizer.param_groups for p in g["params"]}
        torch.cuda.synchronize()
        self.load_seconds=time.perf_counter()-started
        atomic_json(self.run/"parameter_audit.json",dict(strict_load=str(result),original_trainable_parameters=0,
            original_parameters=sum(p.numel() for p in model.parameters()),
            trainable=[dict(name=n,shape=list(p.shape),count=p.numel(),dtype=str(p.dtype)) for n,p in self.head.named_parameters()],
            new_parameter_count=sum(p.numel() for p in self.head.parameters()),original_state_sha=self.frozen_hash,load_seconds=self.load_seconds))

    def forward(self,record,full=False,texture=False,fraction=None):
        from reliability_data import load_pair,label_parents
        start=time.perf_counter()
        atomic_json(self.run/"heartbeat.json",dict(time=now(),event="forward_start",pair_id=record["pair_id"],pid=os.getpid()))
        data=load_pair(record,device="cuda",resolution=self.config["resolution"])
        io=time.perf_counter()-start
        self.model.eval()
        self.model.texture_budget_enabled=bool(texture)
        if fraction is not None:self.model.texture_fraction=float(fraction)
        torch.cuda.synchronize();t=time.perf_counter()
        with torch.no_grad(),torch.cuda.amp.autocast(enabled=self.config["amp"]):
            self.model(data,coarse_only=not full)
        torch.cuda.synchronize()
        for key in ("feat0_c","feat1_c","coarse_confidence","coarse_coord_0","coarse_coord_1"):
            if not torch.isfinite(data[key]).all():raise FloatingPointError("Nonfinite original output: "+key)
        if data["feat0_c"].shape[1]!=self.model.d_coarse or data["feat1_c"].shape[1]!=self.model.d_coarse:
            raise AssertionError("Actual descriptor channels differ from configured original model")
        self.calls["backbone"]+=1
        if full:self.calls["original_fine"]+=1
        data["_timing"]=dict(io=io,backbone_and_matching=time.perf_counter()-t)
        labels=label_parents(data)
        atomic_json(self.run/"heartbeat.json",dict(time=now(),event="forward_complete",pair_id=record["pair_id"],pid=os.getpid()))
        return data,labels

    def logits(self,data):
        self.calls["head_forward"]+=1
        with torch.cuda.amp.autocast(enabled=False):
            value=self.head(data["coarse_descriptor0"].detach().float(),data["coarse_descriptor1"].detach().float(),data["coarse_confidence"].detach().float())
        if not torch.isfinite(value).all():raise FloatingPointError("Nonfinite reliability logit")
        return value.reshape(-1)

    def check_frozen(self):
        if any(p.grad is not None or p.requires_grad for p in self.model.parameters()):raise AssertionError("Original gradient/freeze violation")
        if any(m.training for m in self.model.modules()):raise AssertionError("Original module training/BN mode changed")
        if state_hash(self.model)!=self.frozen_hash:raise AssertionError("Original parameter/BN state changed")
        return True
