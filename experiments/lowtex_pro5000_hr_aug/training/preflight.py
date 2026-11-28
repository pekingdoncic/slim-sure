import os
from .options import parser
from ..utils.run_io import locations


def main():
    args=parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    from .runtime import Guard
    from .worker import preflight
    _,report=locations(args.mode,args.run_id)
    guard=Guard(args.vram_cap_gib)
    try:
        print(preflight(args,{"report":str(report)},guard))
    finally:
        guard.close()


if __name__=="__main__":
    main()
