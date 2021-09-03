from __future__ import print_function
import argparse
import os
import sys
import torch
from scipy import misc
import numpy as np

parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')
parser.add_argument('--in-dataset', default="CIFAR-10", type=str, help='in-distribution dataset')
parser.add_argument('--out-dataset', default="dtd", type=str, help='out-distribution dataset')
parser.add_argument('--method', default='msp', type=str, help='ood detection method')
parser.add_argument('--gpu', default='0', type=str, help='which gpu to use')
parser.add_argument('--random_seed', default=1, type=int, help='The seed used for torch & numpy')
parser.add_argument('--name', required=True, type=str, help='name of experiment')
parser.set_defaults(argument=False)


def get_curve(known, novel):
    tp, fp = dict(), dict()
    fpr_at_tpr95 = dict()

    known.sort()
    novel.sort()

    end = np.max([np.max(known), np.max(novel)])
    start = np.min([np.min(known),np.min(novel)])

    all = np.concatenate((known, novel))
    all.sort()

    num_k = torch.Tensor(known).shape[0]
    num_n = torch.Tensor(novel).shape[0]

    threshold = known[round(0.05 * num_k)]

    tp = -np.ones([num_k+num_n+1], dtype=int)
    fp = -np.ones([num_k+num_n+1], dtype=int)
    tp[0], fp[0] = num_k, num_n
    k, n = 0, 0
    for l in range(num_k+num_n):
        if k == num_k:
            tp[l+1:] = tp[l]
            fp[l+1:] = np.arange(fp[l]-1, -1, -1)
            break
        elif n == num_n:
            tp[l+1:] = np.arange(tp[l]-1, -1, -1)
            fp[l+1:] = fp[l]
            break
        else:
            if novel[n] < known[k]:
                n += 1
                tp[l+1] = tp[l]
                fp[l+1] = fp[l] - 1
            else:
                k += 1
                tp[l+1] = tp[l] - 1
                fp[l+1] = fp[l]

    j = num_k+num_n-1
    for l in range(num_k+num_n-1):
        if all[j] == all[j-1]:
            tp[j] = tp[j+1]
            fp[j] = fp[j+1]
        j -= 1

    fpr_at_tpr95 = np.sum(novel > threshold) / float(num_n)

    return tp, fp, fpr_at_tpr95


def cal_metric(known, novel):
    tp, fp, fpr_at_tpr95 = get_curve(known, novel)
    results = dict()
    mtypes = ['FPR', 'AUROC', 'DTERR', 'AUIN', 'AUOUT']

    results = dict()

    # FPR
    mtype = 'FPR'
    results[mtype] = fpr_at_tpr95

    # AUROC
    mtype = 'AUROC'
    tpr = np.concatenate([[1.], tp/tp[0], [0.]])
    fpr = np.concatenate([[1.], fp/fp[0], [0.]])
    results[mtype] = -np.trapz(1.-fpr, tpr)

    # DTERR
    mtype = 'DTERR'
    results[mtype] = ((tp[0] - tp + fp) / (tp[0] + fp[0])).min()

    # AUIN
    mtype = 'AUIN'
    denom = tp+fp
    denom[denom == 0.] = -1.
    pin_ind = np.concatenate([[True], denom > 0., [True]])
    pin = np.concatenate([[.5], tp/denom, [0.]])
    results[mtype] = -np.trapz(pin[pin_ind], tpr[pin_ind])

    # AUOUT
    mtype = 'AUOUT'
    denom = tp[0]-tp+fp[0]-fp
    denom[denom == 0.] = -1.
    pout_ind = np.concatenate([[True], denom > 0., [True]])
    pout = np.concatenate([[0.], (fp[0]-fp)/denom, [.5]])
    results[mtype] = np.trapz(pout[pout_ind], 1.-fpr[pout_ind])

    # STAT OF ID
    mtype = 'MEAN_IN'
    results[mtype] = np.mean(known)
    mtype = 'STD_IN'
    results[mtype] = np.std(known)

    # STAT OF OOD
    mtype = 'MEAN_OUT'
    results[mtype] = np.mean(novel)
    mtype = 'STD_OUT'
    results[mtype] = np.std(novel)

    return results


def compute_ood_metrics(args):
    print('Natural OOD')
    print('nat_in vs. nat_out')
    
    in_directory = "./evaluation/{name}/{method}/{in_dataset}/".format(name=args.name, method=args.method, in_dataset=args.in_dataset)
    out_directory = "./evaluation/{name}/{method}/{out_dataset}/".format(name=args.name, method=args.method, out_dataset=args.out_dataset)
    known = np.loadtxt(in_directory + "in_scores.txt", delimiter='\n')
    novel = np.loadtxt(out_directory + "out_scores.txt", delimiter='\n')

    known_sorted = np.sort(known)
    num_k = known.shape[0]
    threshold = known_sorted[round(0.05 * num_k)]

    in_cond = (novel>threshold).astype(np.float32)
    results = cal_metric(known, novel)

    mtypes = ['FPR', 'DTERR', 'AUROC', 'AUIN', 'AUOUT']
    print('in_distribution: ' + args.in_dataset)
    print('out_distribution: '+ args.out_dataset)
    print('Model Name: ' + args.name)
    print('')
    print(' OOD detection method: ' + args.method)
    
    for mtype in mtypes:
        print(' {mtype:6s}'.format(mtype=mtype), end='')
    
    print('\n{val:6.2f}'.format(val=100.*results['FPR']), end='')
    print(' {val:6.2f}'.format(val=100.*results['DTERR']), end='')
    print(' {val:6.2f}'.format(val=100.*results['AUROC']), end='')
    print(' {val:6.2f}'.format(val=100.*results['AUIN']), end='')
    print(' {val:6.2f}\n'.format(val=100.*results['AUOUT']), end='')
    print('')

    return

if __name__ == '__main__':
    args = parser.parse_args()
    print_args = '*'*45
    for key,value in args._get_kwargs():
        print_args = print_args + '\n- ' + str(key) + " -> " + str(value)

    print_args = print_args + '\n' + '*'*45
    print(print_args)

    # Setting up gpu parameters
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Setting up random seeds
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    compute_ood_metrics(args)
