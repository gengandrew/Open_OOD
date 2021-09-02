from __future__ import print_function
import argparse
import os
import abc
import sys

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.linear_model import LogisticRegressionCV
import models.densenet as dn
import models.wideresnet as wn
from models.purned_model import PrunedModel
import models.resnet18 as rn
import models.gmm as gmmlib
import numpy as np
import time

parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')
parser.add_argument('--in-dataset', default="CIFAR-10", type=str, help='in-distribution dataset')
parser.add_argument('--out-dataset', default="dtd", type=str, help='out-distribution dataset')
parser.add_argument('--name', required=True, type=str, help='neural network name and training set')
parser.add_argument('--random_seed', default=1, type=int, help='The seed used for torch & numpy')
parser.add_argument('--model-arch', default='resnet_18', type=str, help='model architecture')
parser.add_argument('--gpu', default = '0', type = str, help='gpu index')
parser.add_argument('--method', default='msp', type=str, help='ood detection method')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=50, type=int, help='mini-batch size')

parser.add_argument('--severity-level', default=5, type=int, help='severity level')
parser.add_argument('--layers', default=100, type=int, help='total number of layers (default: 100)')

parser.set_defaults(argument=False)

def get_msp_score(inputs, model):
    with torch.no_grad():
        outputs = model(inputs)
    scores = np.max(F.softmax(outputs, dim=1).detach().cpu().numpy(), axis=1)

    return scores

def get_sofl_score(inputs, model, method_args):
    num_classes = method_args['num_classes']
    with torch.no_grad():
        outputs = model(inputs)
    scores = -F.softmax(outputs, dim=1)[:, num_classes:].sum(dim=1).detach().cpu().numpy()

    return scores

def get_rowl_score(inputs, model, method_args, raw_score=False):
    num_classes = method_args['num_classes']
    with torch.no_grad():
        outputs = model(inputs)

    if raw_score:
        scores = -1.0 * F.softmax(outputs, dim=1)[:, num_classes].float().detach().cpu().numpy()
    else:
        scores = -1.0 * (outputs.argmax(dim=1)==num_classes).float().detach().cpu().numpy()

    return scores

def get_atom_score(inputs, model, method_args):
    num_classes = method_args['num_classes']
    with torch.no_grad():
        outputs = model(inputs)
    #scores = -F.softmax(outputs, dim=1)[:, num_classes]
    scores = -1.0 * (F.softmax(outputs, dim=1)[:,-1]).float().detach().cpu().numpy()

    return scores

def get_odin_score(inputs, model, temperature, magnitude):
    # Calculating the perturbation we need to add, that is,
    # the sign of gradient of cross entropy loss w.r.t. input
    criterion = nn.CrossEntropyLoss()
    inputs = Variable(inputs, requires_grad = True)
    outputs = model(inputs)

    maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)

    # Using temperature scaling
    outputs = outputs / temperature

    labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
    loss = criterion(outputs, labels)
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient =  torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2

    # Adding small perturbations to images
    tempInputs = torch.add(inputs.data,  -magnitude, gradient)
    outputs = model(Variable(tempInputs))
    outputs = outputs / temperature
    # Calculating the confidence after adding perturbations
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
    scores = np.max(nnOutputs, axis=1)

    return scores

def get_mahalanobis_score(inputs, model, method_args):
    num_classes = method_args['num_classes']
    sample_mean = method_args['sample_mean']
    precision = method_args['precision']
    magnitude = method_args['magnitude']
    regressor = method_args['regressor']
    num_output = method_args['num_output']

    Mahalanobis_scores = get_Mahalanobis_score(inputs, model, num_classes, sample_mean, precision, num_output, magnitude)
    scores = -regressor.predict_proba(Mahalanobis_scores)[:, 1]

    return scores

def get_score(inputs, model, method, method_args, raw_score=False):
    if method == "msp":
        scores = get_msp_score(inputs, model, method_args)
    elif method == "odin":
        scores = get_odin_score(inputs, model, method_args)
    elif method == "mahalanobis":
        scores = get_mahalanobis_score(inputs, model, method_args)
    elif method == "sofl":
        scores = get_sofl_score(inputs, model, method_args)
    elif method == "rowl":
        scores = get_rowl_score(inputs, model, method_args, raw_score)
    elif method == "atom":
        scores = get_atom_score(inputs, model, method_args)

    return scores

def corrupt_attack(x, model, method, method_args, in_distribution, severity_level = 5):

    x = x.detach().clone()

    scores = get_score(x, model, method, method_args, raw_score=True)

    worst_score = scores.copy()
    worst_x = x.clone()

    xs = gen_corruction_image(x.cpu(), severity_level)

    for curr_x in xs:
        curr_x = curr_x.cuda()
        scores = get_score(curr_x, model, method, method_args, raw_score=True)

        if in_distribution:
            cond = scores < worst_score
        else:
            cond = scores > worst_score

        worst_score[cond] = scores[cond]
        worst_x[cond] = curr_x[cond]

    return worst_x

def get_detector_hyperparameters(args):
    if args.method == "odin":
        if args.model_arch == 'densenet':
            if args.in_dataset == "CIFAR-10":
                return (1000.0, 0.0016)
            elif args.in_dataset == "CIFAR-100":
                return (1000.0, 0.0012)
            elif args.in_dataset == "SVHN":
                return (1000.0, 0.0006)
        elif args.model_arch == 'resnet_18':
            if args.in_dataset == "CIFAR-10":
                return (1000.0, 0.0006)
            elif args.in_dataset == "CIFAR-100":
                return (1000.0, 0.0012)
            elif args.in_dataset == "SVHN":
                return (1000.0, 0.0002)
        else:
            assert False, 'Not supported model arch'
    
    return temperature, magnitude


def eval_ood_detector(args, directory):
    ########################################In-distribution###########################################
    if args.in_dataset == "CIFAR-10":
        normalizer = transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0))
        testset = torchvision.datasets.CIFAR10(root='./datasets/cifar10', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        testloaderIn = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    num_classes = 10

    if args.model_arch == 'resnet_18':
        model = model = rn.ResNet18()
    elif args.model_arch == 'densenet':
        model = dn.DenseNet3(args.layers, num_classes, normalizer=normalizer)
    else:
        assert False, 'Not supported model arch: {}'.format(args.model_arch)

    checkpoint = torch.load("./trained_models/{in_dataset}/{name}/epoch_{epochs}.pth".format(in_dataset=args.in_dataset, name=args.name, epochs=args.epochs))
    model.load_state_dict(checkpoint)
    model.eval()
    model.cuda()

    print("Processing in-distribution images")

    # Creating In-distribution directory
    in_directory = directory + "{in_dataset}/".format(in_dataset=args.in_dataset)
    if not os.path.exists(in_directory):
        os.makedirs(in_directory)

    in_score_file = open(os.path.join(in_directory, "in_scores.txt"), 'w')
    in_label_file = open(os.path.join(in_directory, "in_labels.txt"), 'w')

    N = len(testloaderIn.dataset)
    count = 0
    for j, data in enumerate(testloaderIn):
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        curr_batch_size = images.shape[0]

        inputs = images
        if args.method == "odin":
            (temperature, magnitude) = get_detector_hyperparameters(args)
            scores = get_odin_score(inputs, model, temperature, magnitude)
        else:
            scores = get_msp_score(inputs, model)

        for score in scores:
            in_score_file.write("{}\n".format(score))

        outputs = F.softmax(model(inputs)[:, :num_classes], dim=1)
        outputs = outputs.detach().cpu().numpy()
        preds = np.argmax(outputs, axis=1)
        confs = np.max(outputs, axis=1)

        for k in range(preds.shape[0]):
            in_label_file.write("{} {} {}\n".format(labels[k], preds[k], confs[k]))

        count += curr_batch_size
        print("{:4}/{:4} images processed".format(count, N))

    in_score_file.close()
    in_label_file.close()

    ###################################Out-of-Distributions#####################################
    print("Processing out-of-distribution images")

    # Creating Out-distribution directory
    out_directory = directory + "{out_dataset}/".format(out_dataset=args.out_dataset)
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)

    out_score_file = open(os.path.join(out_directory, "out_scores.txt"), 'w')

    if args.out_dataset == 'SVHN':
        testsetout = svhn.SVHN('datasets/ood_datasets/svhn/', split='test', transform=transforms.ToTensor(), download=False)
        testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size, shuffle=True, num_workers=2)
    elif args.out_dataset == 'dtd':
        testsetout = torchvision.datasets.ImageFolder(root="datasets/ood_datasets/dtd/images",
                                    transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()]))
        testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size, shuffle=True, num_workers=2)
    elif args.out_dataset == 'places365':
        testsetout = torchvision.datasets.ImageFolder(root="datasets/ood_datasets/places365/test_subset",
                                    transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()]))
        testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size, shuffle=True, num_workers=2)
    else:
        testsetout = torchvision.datasets.ImageFolder("./datasets/ood_datasets/{}".format(args.out_dataset),
                                    transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()]))
        testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size, shuffle=True, num_workers=2)

    N = len(testloaderOut.dataset)
    count = 0
    for j, data in enumerate(testloaderOut):
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        curr_batch_size = images.shape[0]

        inputs = images
        if args.method == "odin":
            (temperature, magnitude) = get_detector_hyperparameters(args)
            scores = get_odin_score(inputs, model, temperature, magnitude)
        else:
            scores = get_msp_score(inputs, model)

        for score in scores:
            out_score_file.write("{}\n".format(score))

        count += curr_batch_size
        print("{:4}/{:4} images processed".format(count, N))

    out_score_file.close()
    return


if __name__ == '__main__':
    # Grabbing cli arguments and printing results
    args = parser.parse_args()
    print_args = '*'*45
    for key,value in args._get_kwargs():
        print_args = print_args + '\n- ' + str(key) + " -> " + str(value)

    print_args = print_args + '\n' + '*'*45
    print(print_args)

    # Creating checkpoint directory
    directory = "./evaluation/{name}/{method}/".format(name=args.name, method=args.method)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Saving config args to checkpoint directory
    save_state_file = os.path.join(directory, 'args.txt')
    fw = open(save_state_file, 'w')
    print(print_args, file=fw)
    fw.close()

    # Setting up gpu parameters
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Setting up random seeds
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    eval_ood_detector(args, directory)
