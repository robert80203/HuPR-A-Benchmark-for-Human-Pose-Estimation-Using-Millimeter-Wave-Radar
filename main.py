import argparse
import yaml
from collections import namedtuple
from tools import Trainer, RFTrainer, Runner


class obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, obj(b) if isinstance(b, dict) else b)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')
    parser.add_argument('--dir', type=str, default='test', metavar='B',
                        help='directory of saving/loading')
    parser.add_argument('--visDir', type=str, default='test', metavar='B',
                        help='directory of visualization')
    parser.add_argument('--config', type=str, default='multichirps.yaml', metavar='B',
                        help='directory of visualization')
    parser.add_argument('--gpuIDs', default=[0], type=eval, help='IDs of GPUs to use')                        
    parser.add_argument('--eval', action="store_true")
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--vis', action="store_true", help='visualize the results')
    parser.add_argument('-sr', '--sampling_ratio', type=int, default=1, help='sampling ratio for training/test (default: 1)')
    parser.add_argument('--setting', type=str, default='eccv2022', help='should be prgcn/eccv2022/rfpose')
    args = parser.parse_args()
    with open('./config/' + args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        cfg = obj(cfg)
    if args.setting == 'prgcn':
        trigger = Trainer(args, cfg)
    elif args.setting == 'eccv2022':
        trigger = Runner(args, cfg)
    elif args.setting == 'rfpose':
        trigger = RFTrainer(args, cfg)
    else:
        exit(0)
    
    if args.eval:
        trigger.loadModelWeight('model_best')
        if "stage2" in cfg.MODEL.frontModel:
            trigger.evalRefine(visualization=args.vis)
        else:
            trigger.eval(visualization=args.vis)
    else:
        trigger.loadModelWeight('checkpoint')
        if "stage2" in cfg.MODEL.frontModel:
            trigger.trainRefine()
        else:
            trigger.train()
        #trigger.debug()