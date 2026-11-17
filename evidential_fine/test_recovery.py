"""Crash-boundary recovery check; synthetic journals only, no model or optimizer."""
import os,sys,tempfile
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evidential_fine.common import *
from evidential_fine.training import reconcile_blocks
def run(root):
    root=Path(root);(root/'checks').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root/'checks',prefix='recovery_check_') as td:
        p=Path(td);first=[dict(epoch=0,position=i,updated=i%9!=0) for i in range(50)]
        second=[dict(epoch=0,position=i,updated=True) for i in range(50,100)]
        a=p/'blocks/epoch_00_pos_000000_000049.json';b=p/'blocks/epoch_00_pos_000050_000099.json'
        save(a,first);save(b,second);hasha=sha(a);hashb=sha(b)
        state=dict(visits=50,updates=sum(x['updated'] for x in first));result=reconcile_blocks(p,state)
        assert sha(a)==hasha and not b.exists()
        preserved=Path(result['uncommitted_preserved'][0]['preserved']);assert sha(preserved)==hashb
        assert not reconcile_blocks(p,state)['uncommitted_preserved']
        # A missing committed record is corruption, never silently accepted.
        broken=first[1:];save(a,broken);rejected=False
        try:reconcile_blocks(p,state)
        except AssertionError:rejected=True
        assert rejected
    result=dict(status='pass',time_beijing=now(),interrupted_block_preserved=True,committed_block_unchanged=True,idempotent_reconcile=True,committed_gap_rejected=True,
        scope='Synthetic crash boundary only; real model/optimizer/RNG resume tested by E0 update199->200')
    save(root/'recovery_contract_checks.json',result);print(json.dumps(result))
if __name__=='__main__':run(sys.argv[1])
