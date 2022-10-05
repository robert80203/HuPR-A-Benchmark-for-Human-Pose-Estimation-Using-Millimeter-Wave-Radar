import yaml
import argparse
from tools import Runner
from collections import namedtuple


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
    parser.add_argument('--visDir', type=str, default='none', metavar='B',
                        help='directory of visualization')
    parser.add_argument('--config', type=str, default='20211106/test.yaml', metavar='B',
                        help='directory of visualization')
    parser.add_argument('--gpuIDs', default=[0], type=eval, help='IDs of GPUs to use')                        
    parser.add_argument('--eval', action="store_true")
    parser.add_argument('-sr', '--sampling_ratio', type=int, default=1, help='sampling ratio for training/test (default: 1)')
    parser.add_argument('--keypoints', action='store_true', help='print out the APs of all keypoints')
    args = parser.parse_args()
    with open('./config/' + args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        cfg = obj(cfg)
    trigger = Runner(args, cfg)
    vis = False if args.visDir == 'none' else True
    if args.eval:
        trigger.loadModelWeight('model_best')
        trigger.eval(visualization=vis)
    else:
        trigger.loadModelWeight('checkpoint')
        trigger.train()