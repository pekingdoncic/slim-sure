"""Atomic per-visit scalar records and cumulative pair-equal streamed L-BFGS."""
import json,os,sqlite3,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from .runtime import atomic_json,sha,digest

def scalar_features(mconf,norm0,norm1):
    x=np.stack([mconf,np.log1p(norm0),np.log1p(norm1)],axis=-1).astype(np.float32)
    if x.ndim!=2 or x.shape[1]!=3 or not np.isfinite(x).all():raise ValueError("Invalid scalar inputs")
    return x

def normalize(x,stats):
    return (np.asarray(x,np.float64)-np.asarray(stats["mean"]))/np.asarray(stats["scale"])

class ScalarStore:
    def __init__(self,root,identity,readonly=False):
        self.root=Path(root);self.identity=identity;self.identity_sha=digest(identity);self.readonly=readonly
        if not readonly:self.root.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect("file:"+str(self.root/"records.sqlite")+"?mode=ro",uri=True) if readonly else sqlite3.connect(self.root/"records.sqlite")
        self.db.execute("PRAGMA cache_size=-4096")
        if not readonly:
            self.db.executescript("PRAGMA journal_mode=WAL;PRAGMA synchronous=FULL;CREATE TABLE IF NOT EXISTS blocks(epoch INTEGER,position INTEGER,pair_id TEXT,direction TEXT,path TEXT,sha TEXT,n INTEGER,identity TEXT,PRIMARY KEY(epoch,position));")
            self.db.commit()
    def commit(self,epoch,position,pair_id,direction,raw_ids,x,y):
        x=np.asarray(x,np.float32);y=np.asarray(y,np.float32);raw=np.asarray(raw_ids,np.int64)
        if not len(y) or x.shape!=(len(y),3) or raw.shape!=y.shape or len(np.unique(raw))!=len(raw):raise ValueError("Bad shared sample")
        if not np.isfinite(x).all() or not np.isin(y,[0,1]).all():raise ValueError("Nonfinite or nonbinary scalar record")
        key=(int(epoch),int(position));old=self.db.execute("SELECT pair_id,direction,path,sha,identity FROM blocks WHERE epoch=? AND position=?",key).fetchone()
        if old:
            if (old[0],old[1],old[4])!=(pair_id,direction,self.identity_sha):raise ValueError("Visit identity changed")
            path=self.root/old[2]
            if sha(path)!=old[3]:raise ValueError("Scalar block checksum changed")
            with np.load(path) as z:
                if not np.array_equal(raw,z["raw_id"]) or not np.array_equal(y,z["y"]) or not np.allclose(x,z["x"],atol=1e-5,rtol=1e-5):raise ValueError("Replayed scalar sample differs")
            return dict(reused=True,n=len(y),sha256=old[3])
        name=f"epoch_{epoch:02d}_visit_{position:06d}.npz";path=self.root/name;tmp=path.with_suffix(".tmp")
        with tmp.open("wb") as f:
            np.savez(f,x=x,y=y,raw_id=raw,weight=np.full(len(y),1./len(y),np.float32));f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
        directory=os.open(self.root,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)
        h=sha(path)
        self.db.execute("INSERT INTO blocks VALUES(?,?,?,?,?,?,?,?)",(*key,pair_id,direction,name,h,len(y),self.identity_sha));self.db.commit()
        return dict(reused=False,n=len(y),sha256=h,bytes=path.stat().st_size)
    def blocks(self,through_epoch):
        for epoch,pos,pid,direction,name,h,n,ident in self.db.execute("SELECT * FROM blocks WHERE epoch<=? ORDER BY epoch,position",(through_epoch,)):
            if ident!=self.identity_sha:raise ValueError("Scalar store identity changed")
            path=self.root/name
            # Blocks are immutable; checksum is checked before each fit, not every line search.
            with np.load(path) as z:yield z["x"],z["y"],z["weight"]
    def signature(self,through_epoch):
        import hashlib
        h=hashlib.sha256();n=visits=total_bytes=0
        for row in self.db.execute("SELECT epoch,position,pair_id,direction,path,sha,n,identity FROM blocks WHERE epoch<=? ORDER BY epoch,position",(through_epoch,)):
            if sha(self.root/row[4])!=row[5]:raise ValueError("Scalar block checksum changed")
            h.update(json.dumps(row,separators=(",",":")).encode());visits+=1;n+=row[6];total_bytes+=(self.root/row[4]).stat().st_size
        return dict(sha256=h.hexdigest(),visits=visits,parents=n,bytes=total_bytes)
    def close(self):self.db.close()

def fit_cumulative(store,through_epoch,stats,out,initial=None,maxiter=500):
    signature=store.signature(through_epoch)
    if not signature["visits"]:raise ValueError("No committed supervised visits for LR")
    source=dict(records=signature,standardization_sha256=digest(stats),epoch=through_epoch,l2=1e-6,
                loss="sum per-visit mean BCE / supervised visits + 0.5e-6 * squared coefficient norm; intercept unregularized")
    target=Path(out)
    if target.exists():
        saved=json.loads(target.read_text())
        if saved["source"]!=source:raise ValueError("LR fit source changed")
        if saved.get("status")=="committed" and saved.get("converged"):return saved
        # A failed fit is evidence, never a reusable successful phase boundary.
    evaluations=0
    def objective(theta):
        nonlocal evaluations
        evaluations+=1
        atomic_json(target.with_suffix(".progress.json"),dict(epoch=through_epoch,evaluation=evaluations,time=time.time(),stage="lr_fit"))
        loss=0.;grad=np.zeros(4,np.float64)
        for block_position,(x,y,w) in enumerate(store.blocks(through_epoch)):
            if block_position%1024==0:
                atomic_json(target.with_suffix(".progress.json"),dict(epoch=through_epoch,evaluation=evaluations,
                    block_position=block_position,total_visits=signature["visits"],time=time.time(),stage="lr_fit"))
            z=normalize(x,stats);y=y.astype(np.float64);w=w.astype(np.float64)
            # Re-normalize the stored float32 weights within a visit so each visit totals exactly one.
            w=w/w.sum();logit=z@theta[:3]+theta[3]
            loss+=float(np.dot(w,np.logaddexp(0,logit)-y*logit))
            residual=w*(expit(logit)-y);grad[:3]+=z.T@residual;grad[3]+=residual.sum()
        loss/=signature["visits"];grad/=signature["visits"]
        loss+=.5e-6*np.dot(theta[:3],theta[:3]);grad[:3]+=1e-6*theta[:3]
        return loss,grad
    started=time.monotonic()
    result=minimize(objective,np.zeros(4) if initial is None else np.asarray(initial,np.float64),jac=True,
                    method="L-BFGS-B",options=dict(maxiter=maxiter,gtol=1e-7,ftol=1e-12,maxls=50))
    loss,gradient=objective(result.x)
    value=dict(status="committed" if result.success else "blocked",source=source,parameters=result.x.tolist(),
               converged=bool(result.success),iterations=int(result.nit),evaluations=int(result.nfev),loss=float(loss),
               gradient_inf_norm=float(np.abs(gradient).max()),message=str(result.message),seconds=time.monotonic()-started,
               convergence=dict(gtol=1e-7,ftol=1e-12,maxiter=maxiter,maxls=50),identity=store.identity)
    atomic_json(target,value)
    if not result.success:raise RuntimeError("Cumulative LR did not converge: "+str(result.message))
    return value

def lr_logits(x,stats,fit):
    theta=np.asarray(fit["parameters"]);return normalize(x,stats)@theta[:3]+theta[3]
