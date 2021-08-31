from __future__ import print_function
import argparse
import os
import sys
import torch
from scipy import misc
import numpy as np

parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')
parser.add_argument('--in-dataset', default="CIFAR-10", type=str, help='in-distribution dataset')
parser.add_argument('--saved_form', default="list", type=str, help='format of saved result')
parser.add_argument('--base-dir', default='output/ood_scores', type=str, help='result directory')
parser.add_argument('--epsilon', default=8, type=int, help='epsilon')
parser.add_argument('--gpu', default='0', type=str, help='which gpu to use')
parser.add_argument('--dataset', default="dtd", type=str, help='distribution dataset')
parser.add_argument('--score', default="min", type=str, help='scoring function used')
parser.add_argument('--name', required=True, type=str, help='name of experiment')
parser.set_defaults(argument=True)

np.random.seed(1)

def parse_autoencoder_single_scores(name, dataset, index):
    path = "./gradients/{name}/gmm/level_0/{dataset}_score.pth".format(name=name, dataset=dataset)
    data = torch.load(path)

    parsed_data = np.array(data["linear.weight"])
    return -parsed_data


def parse_gmm_single_scores(name, dataset, index):
    def get_data_from_load(unparsed_data, index):
        parsed_data = []
        names = [name for name, score_list in unparsed_data.items()]

        for score in unparsed_data[names[index]]:
            parsed_data.append(score.item())
        
        return np.array(parsed_data)
    
    path = "./gradients/{name}/gmm/level_0/{dataset}_score.pth".format(name=name, dataset=dataset)
    data = torch.load(path)

    parsed_data = get_data_from_load(data, index)

    return parsed_data

def parse_gmm_scores(index):
    def get_data_from_load(unparsed_data, index):
        parsed_data = []
        names = [name for name, score_list in unparsed_data.items()]

        for score in unparsed_data[names[index]]:
            parsed_data.append(score.item())
        
        return np.array(parsed_data)
    
    indistribution = torch.load('./gradients/resnet_bce/gmm/level_0/CIFAR-10_score.pth')
    dtd = torch.load('./gradients/resnet_bce/gmm/level_0/dtd_score.pth')
    iSUN = torch.load('./gradients/resnet_bce/gmm/level_0/iSUN_score.pth')
    LSUN = torch.load('./gradients/resnet_bce/gmm/level_0/LSUN_score.pth')
    LSUN_resize = torch.load('./gradients/resnet_bce/gmm/level_0/LSUN_resize_score.pth')

    in_data = get_data_from_load(indistribution, index)
    dtd_data = get_data_from_load(dtd, index)
    iSUN_data = get_data_from_load(iSUN, index)
    LSUN_data = get_data_from_load(LSUN, index)
    LSUN_resize_data = get_data_from_load(LSUN_resize, index)

    return (in_data, dtd_data, iSUN_data, LSUN_data, LSUN_resize_data)

def parse_original_form(index):
    def get_data_from_load(unparsed_data):
        parsed_data_dict = dict()

        for data_iteration in unparsed_data:
            for name, L2_norm in data_iteration.items():
                if name not in parsed_data_dict:
                    parsed_data_dict[name] = np.array([])
                
                parsed_data_dict[name] = np.append(parsed_data_dict[name], L2_norm.cpu().item())

        names = [name for name, data in parsed_data_dict.items()]
        return (names, parsed_data_dict)
    
    indistribution = torch.load('./gradients_original/resnet_bce/raw/level_0/CIFAR-10.pth')
    dtd = torch.load('./gradients_original/resnet_bce/raw/level_0/dtd.pth')
    iSUN = torch.load('./gradients_original/resnet_bce/raw/level_0/iSUN.pth')
    LSUN = torch.load('./gradients_original/resnet_bce/raw/level_0/LSUN.pth')
    LSUN_resize = torch.load('./gradients_original/resnet_bce/raw/level_0/LSUN_resize.pth')

    (names, in_data) = get_data_from_load(indistribution)
    (names, dtd_data) = get_data_from_load(dtd)
    (names, iSUN_data) = get_data_from_load(iSUN)
    (names, LSUN_data) = get_data_from_load(LSUN)
    (names, LSUN_resize_data) = get_data_from_load(LSUN_resize)

    return (in_data[names[index]], dtd_data[names[index]], iSUN_data[names[index]], LSUN_data[names[index]], LSUN_resize_data[names[index]])


def parse_original_single_form(dirname, dataset, index):
    def get_data_from_load(unparsed_data):
        parsed_data_dict = dict()

        for data_iteration in unparsed_data:
            for name, L2_norm in data_iteration.items():
                if name not in parsed_data_dict:
                    parsed_data_dict[name] = np.array([])
                
                parsed_data_dict[name] = np.append(parsed_data_dict[name], L2_norm)

        names = [name for name, data in parsed_data_dict.items()]
        return (names, parsed_data_dict)
    
    distribution = torch.load('./gradients_original/{dirname}/raw/level_0/{dataset}.pth'.format(dirname=dirname, dataset=dataset))

    (names, data) = get_data_from_load(distribution)

    return data[names[index]]


def parse_gradient_L2_dict(name, score, dataset, index):
    def get_data_from_load(unparsed_data, index):
        parsed_data = []
        names = [name for name, score_list in unparsed_data.items()]

        for score in unparsed_data[names[index]]:
            parsed_data.append(score)
        
        return np.array(parsed_data)
    
    path = "./gradients/{name}/{score}/level_0/{dataset}.pth".format(name=name, score=score, dataset=dataset)
    data = torch.load(path)
    parsed_data = get_data_from_load(data, index)

    return -parsed_data


def parse_energy_single_dict(name, dataset):    
    path = "./gradients_energy/{name}/raw/level_0/{dataset}.pth".format(name=name, dataset=dataset)
    data = torch.load(path)
    data = np.array(data)

    return -data


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

    return results

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

def print_results(results):
    mtypes = ['FPR', 'DTERR', 'AUROC', 'AUIN', 'AUOUT']

    print(' OOD detection using Gradient Tracking ')
    for mtype in mtypes:
        print(' {mtype:6s}'.format(mtype=mtype), end='')
    
    print('\n{val:6.2f}'.format(val=100.*results['FPR']), end='')
    print(' {val:6.2f}'.format(val=100.*results['DTERR']), end='')
    print(' {val:6.2f}'.format(val=100.*results['AUROC']), end='')
    print(' {val:6.2f}'.format(val=100.*results['AUIN']), end='')
    print(' {val:6.2f}\n'.format(val=100.*results['AUOUT']), end='')
    print('')

def compute_average_results(all_results):
    mtypes = ['FPR', 'DTERR', 'AUROC', 'AUIN', 'AUOUT']
    avg_results = dict()

    for mtype in mtypes:
        avg_results[mtype] = 0.0

    for results in all_results:
        for mtype in mtypes:
            avg_results[mtype] += results[mtype]

    for mtype in mtypes:
        avg_results[mtype] /= float(len(all_results))

    return avg_results

def compute_traditional_ood(all_dataset):
    print('Natural OOD')
    print('nat_in vs. nat_out')

    in_distribution = all_dataset[0]
    known = np.array(in_distribution)
    known_sorted = np.sort(known)
    num_k = len(known)

    threshold = known_sorted[round(0.05 * num_k)]

    all_results = []
    total = 0.0

    for novel in all_dataset[1::]:
        novel = np.array(novel)
        in_cond = (novel>threshold).astype(np.float32)
        total += torch.Tensor(novel).shape[0]

        results = cal_metric(known, novel)
        all_results.append(results)

    avg_results = compute_average_results(all_results)
    print_results(avg_results)

    return


if __name__ == '__main__':
    args = parser.parse_args()
    print_args = '*'*45
    for key,value in args._get_kwargs():
        print_args = print_args + '\n- ' + str(key) + " -> " + str(value)

    print_args = print_args + '\n' + '*'*45
    print(print_args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.saved_form == 'norm_single':
        for index in range(0, 20):
            print("Evaluation at layer index [{index}]".format(index=index))
            in_data = parse_gradient_L2_dict(args.name, args.score, "CIFAR-10", index)
            out_data = parse_gradient_L2_dict(args.name, args.score, args.dataset, index)
            all_dataset = [in_data, out_data]
            compute_traditional_ood(all_dataset)
    elif args.saved_form == 'original':
        for index in range(0, 20):
            print("Evaluation at layer index [{index}]".format(index=index))
            (in_distribution, dtd, iSUN, LSUN, LSUN_resize) = parse_original_form(index)
            compute_traditional_ood(in_distribution, dtd, iSUN, LSUN, LSUN_resize)
    elif args.saved_form == 'original_single':
        for index in range(0, 20):
            print("Evaluation at layer index [{index}]".format(index=index))
            in_data = parse_original_single_form(args.name, "CIFAR-10", index)
            out_data = parse_original_single_form(args.name, args.dataset, index)
            all_dataset = [in_data, out_data]
            compute_traditional_ood(all_dataset)
    elif args.saved_form == 'gmm':
        for index in range(0, 20):
            print("Evaluation at layer index [{index}]".format(index=index))
            (in_distribution, dtd, iSUN, LSUN, LSUN_resize) = parse_gmm_scores(index)
            all_dataset = [in_distribution, dtd, iSUN, LSUN, LSUN_resize]
            compute_traditional_ood(all_dataset)
    elif args.saved_form == 'gmm_single':
        for index in range(0, 20):
            print("Evaluation at layer index [{index}]".format(index=index))
            in_dataset = parse_gmm_single_scores(args.name, "CIFAR-10", index)
            out_dataset = parse_gmm_single_scores(args.name, args.dataset, index)
            all_dataset = [in_dataset, out_dataset]
            compute_traditional_ood(all_dataset)
    elif args.saved_form == 'encoder_single':
        for index in range(0, 20):
            print("Evaluation at layer index [{index}]".format(index=index))
            in_dataset = parse_autoencoder_single_scores(args.name, "CIFAR-10", index)
            out_dataset = parse_autoencoder_single_scores(args.name, args.dataset, index)
            all_dataset = [in_dataset, out_dataset]
            compute_traditional_ood(all_dataset)
    elif args.saved_form == 'energy_single':
        print("Evaluation of energy_single for {dataset}".format(dataset=args.dataset))
        in_dataset = parse_energy_single_dict(args.name, "CIFAR-10")
        out_dataset = parse_energy_single_dict(args.name, args.dataset)
        all_dataset = [in_dataset, out_dataset]
        compute_traditional_ood(all_dataset)