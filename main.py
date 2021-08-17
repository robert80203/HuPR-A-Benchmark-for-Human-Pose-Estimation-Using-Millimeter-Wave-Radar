import argparse
import yaml
from collections import namedtuple
from tools import Trainer


class obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, obj(b) if isinstance(b, dict) else b)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    #parser.add_argument('--batchSize', type=int, default=1, metavar='N',
    #                    help='input batch size for training (default: 1)')
    #parser.add_argument('--epochs', type=int, default=200, metavar='N',
    #                    help='number of epochs to train (default: 200)')
    #parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
    #                    help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')
    parser.add_argument('--saveDir', type=str, default='test', metavar='B',
                        help='directory of saving')
    parser.add_argument('--loadDir', type=str, default='None', metavar='B',
                        help='directory of loading')
    parser.add_argument('--visDir', type=str, default='test', metavar='B',
                        help='directory of visualization')
    parser.add_argument('--config', type=str, default='multichirps.yaml', metavar='B',
                        help='directory of visualization')
    #parser.add_argument('--dataset', type=str, default='multiChirps', help='multiFrames/multiChirps', metavar='B')
    parser.add_argument('--gpuIDs', default=[0], type=eval, help='IDs of GPUs to use')                        
    #parser.add_argument('--numWorkers', default=2, type=int, help='Number of data loader threads')
    #parser.add_argument('--dataDir', type=str, default='./data/20210628', help='Path to data directory.')
    parser.add_argument('--eval', action="store_true")
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--vis', action="store_true", help='visualize the results')
    args = parser.parse_args()
    with open('./config/' + args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        cfg = obj(cfg)
    trigger = Trainer(args, cfg)
    
    if args.eval:
        trigger.loadModelWeight()
        trigger.eval(visualization=args.vis)
    else:
        if args.loadDir != 'None':
            trigger.loadModelWeight()
        trigger.train()