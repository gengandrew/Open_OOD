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
import matplotlib.pyplot as plt
from random import randrange

from utils import get_Mahalanobis_score
from tune_mahalanobis import tune_mahalanobis_hyperparams

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
parser.add_argument('--layers', default=100, type=int, help='total number of layers (default: 100)')
parser.set_defaults(argument=False)

def shift_distribution(inputs, shift):
    inputs = inputs.cpu()
    if shift == 'rotation':
        rotater = transforms.RandomRotation(degrees=(0, 180))
        rotated_inputs = [transforms.ToTensor()(rotater(transforms.ToPILImage()(input))) for input in inputs]
        rotated_inputs = torch.stack(rotated_inputs)
        return rotated_inputs.cuda()
    elif shift == 'flip':
        horizontalflipper = transforms.RandomHorizontalFlip(p=0.5)
        verticalflipper = transforms.RandomVerticalFlip(p=0.5)
        flipped_inputs = [transforms.ToTensor()(verticalflipper(horizontalflipper(transforms.ToPILImage()(input)))) for input in inputs]
        flipped_inputs = torch.stack(flipped_inputs)
        return flipped_inputs.cuda()
    elif shift == 'crop':
        cropper = transforms.RandomCrop(size=(24, 24))
        padder = transforms.Pad(padding=8)
        cropped_inputs = [transforms.ToTensor()(padder(cropper(transforms.ToPILImage()(input)))) for input in inputs]
        cropped_inputs = torch.stack(cropped_inputs)
        return cropped_inputs.cuda()
    elif shift == 'rfc':
        rotater = transforms.RandomRotation(degrees=(0, 180))
        horizontalflipper = transforms.RandomHorizontalFlip(p=0.5)
        verticalflipper = transforms.RandomVerticalFlip(p=0.5)
        cropper = transforms.RandomCrop(size=(24, 24))
        padder = transforms.Pad(padding=8)

        rfc_inputs = []
        for input in inputs:
            pli_input = transforms.ToPILImage()(input)
            pli_input = rotater(pli_input)
            pli_input = verticalflipper(horizontalflipper(pli_input))
            pli_input = padder(cropper(pli_input))
            result_input = transforms.ToTensor()(pli_input)
            rfc_inputs.append(result_input)

        rfc_inputs = torch.stack(rfc_inputs)
        return rfc_inputs.cuda()
    else:
        assert False, 'Not supported distribution shift: {}'.format(shift)

def get_msp_score(inputs, model):
    with torch.no_grad():
        outputs = model(inputs)
    scores = np.max(F.softmax(outputs, dim=1).detach().cpu().numpy(), axis=1)

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

    if args.method == 'mahalanobis':
        sample_mean, precision, regressor, magnitude = tune_mahalanobis_hyperparams(args.name, args.in_dataset, args.model_arch)
        # regressor = LogisticRegressionCV(cv=2).fit([[0,0,0,0],[0,0,0,0],[1,1,1,1],[1,1,1,1]],[0,0,1,1])
        # regressor.coef_ = lr_weights
        # regressor.intercept_ = lr_bias

        temp_x = torch.rand(2,3,32,32)
        temp_x = Variable(temp_x).cuda()
        temp_list = model.feature_list(temp_x)[1]
        num_output = len(temp_list)

        mahalanobis_args = dict()
        mahalanobis_args['num_classes'] = num_classes
        mahalanobis_args['num_output'] = num_output
        mahalanobis_args['sample_mean'] = sample_mean
        mahalanobis_args['precision'] = precision
        mahalanobis_args['magnitude'] = magnitude
        mahalanobis_args['regressor'] = regressor

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
        elif args.method == 'mahalanobis':
            scores = get_mahalanobis_score(inputs, model, mahalanobis_args)
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

    if args.out_dataset == 'rotation' or args.out_dataset == 'flip' or args.out_dataset == 'crop' or args.out_dataset == 'rfc':
        testsetout = torchvision.datasets.CIFAR10(root='./datasets/cifar10', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size, shuffle=True, num_workers=2)
    elif args.out_dataset == 'snow':
        ood_directory = "./datasets/ood_datasets/CIFAR-10-C/"
        testsetout = torch.tensor(np.transpose(np.load(os.path.join(ood_directory, args.out_dataset + '.npy')), (0,3,1,2)))
        testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size, shuffle=True, num_workers=2)
        print(testloaderOut)
    elif args.out_dataset == 'SVHN':
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
        if args.out_dataset == 'rotation' or args.out_dataset == 'flip' or args.out_dataset == 'crop' or args.out_dataset == 'rfc':
            inputs = shift_distribution(inputs, args.out_dataset)

        if args.method == "odin":
            (temperature, magnitude) = get_detector_hyperparameters(args)
            scores = get_odin_score(inputs, model, temperature, magnitude)
        elif args.method == 'mahalanobis':
            scores = get_mahalanobis_score(inputs, model, mahalanobis_args)
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
