import os
import abc
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import models.resnet as rn
import torchvision.datasets as datasets
from models.purned_model import PrunedModel
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')
parser.add_argument('--gpu', default='0', type=str, help='which gpu to use')
parser.add_argument('--dataset', default="CIFAR-10", type=str, help='distribution dataset')
parser.add_argument('--saved_form', default="gmm", type=str, help='format of saved result')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    help='batch size (default: 128)')
parser.add_argument('--name', required=True, type=str,
                    help='name of experiment')
parser.add_argument('--pruning_level', default=0, type=int,
                    help='total iterations of pruning')


def get_dataset(args):
    transform = transforms.Compose([transforms.ToTensor()])
    # testset = datasets.CIFAR10(root='./datasets/cifar10', train=False, download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    trainset = datasets.CIFAR10('./datasets/cifar10', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    return trainloader


def extract_gradients(pruned_model):
    model_grad_dict = dict()
    for name, param in pruned_model.model.named_parameters():
        if hasattr(pruned_model, PrunedModel.to_mask_name(name)):
            pruned_grad = param.grad * getattr(pruned_model, PrunedModel.to_mask_name(name))
            model_grad_dict[name] = pruned_grad
        else:
            model_grad_dict[name] = param.grad

    return model_grad_dict


def get_gradient_mean_dict(pruned_model, trainloader):
    criterion = nn.BCEWithLogitsLoss().cuda()
    gradient_divergence_dict = dict()

    for i, (inputs, targets) in enumerate(tqdm(trainloader)):
        if i > 10000:
            break

        inputs = inputs.cuda()
        targets = torch.ones((targets.shape[0], 10)).cuda()
        outputs = pruned_model(inputs)

        pruned_model.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()

        gradient_dict = extract_gradients(pruned_model)

        for name, gradient in gradient_dict.items():
            if 'layer3' in name:
                continue

            if 'bn' not in name and 'bias' not in name:
                if name not in gradient_divergence_dict:
                    gradient_divergence_dict[name] = torch.zeros(gradient.shape).cuda()

                if torch.isnan(gradient).any():
                    print("There is Nan!!!")
                    gradient[torch.isnan(gradient)] = 0
                
                gradient_divergence_dict[name] += gradient
    
    for name, gradient_divergence in gradient_divergence_dict.items():
        gradient_divergence_dict[name] = gradient_divergence_dict[name] / len(trainloader)
    
    return gradient_divergence_dict


def get_gradient_covariance_dict(pruned_model, trainloader, gradient_mean_dict):
    criterion = nn.BCEWithLogitsLoss().cuda()
    gradient_covariance_dict = dict()
    for i, (inputs, targets) in enumerate(tqdm(trainloader)):
        if i > 10000:
            break

        inputs = inputs.cuda()
        targets = torch.ones((targets.shape[0], 10)).cuda()
        outputs = pruned_model(inputs)

        pruned_model.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()

        gradient_dict = extract_gradients(pruned_model)

        for name, gradient in gradient_dict.items():
            # and 'layer3.0.conv1.weight' not in name
            if 'layer3' in name:
                continue

            if 'bn' not in name and 'bias' not in name:
                if torch.isnan(gradient).any():
                    print("There is Nan!!!")
                    gradient[torch.isnan(gradient)] = 0
                
                x_mean = gradient - gradient_mean_dict[name]

                if name not in gradient_covariance_dict:
                    gradient_covariance_dict[name] = torch.ger(torch.flatten(x_mean), torch.flatten(x_mean)).cuda()
                else:
                    gradient_covariance_dict[name] += torch.ger(torch.flatten(x_mean), torch.flatten(x_mean))
    
    for name, gradient_covariance in gradient_covariance_dict.items():
        gradient_covariance_dict[name] = gradient_covariance_dict[name] / len(trainloader)
    
    return gradient_covariance_dict


def nearest_PSD(covariance):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """
    A = covariance.cpu().numpy()

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if check_PSD(A3):
        nearest_PSD = torch.Tensor(A3).cuda()
        return nearest_PSD

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not check_PSD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    nearest_PSD = torch.Tensor(A3).cuda()
    return nearest_PSD


def check_PSD(matrix):
    try:
        _ = torch.cholesky(torch.Tensor(matrix))
        return True
    except:
        return False


def fix_not_PSD_covariance(gmm_covariance):
    print("# ------------------------ Altering Covariance to PSD ------------------------ #")
    for name, covariance in tqdm(gmm_covariance.items()):
        if not check_PSD(gmm_covariance[name]):
            gmm_covariance[name] += (0.00000000001 * torch.rand(gmm_covariance[name].shape).cuda())
            gmm_covariance[name] = nearest_PSD(gmm_covariance[name])
    
    return gmm_covariance


if __name__ == '__main__':
    args = parser.parse_args()
    print_args = '*'*45
    for key,value in args._get_kwargs():
        print_args = print_args + '\n- ' + str(key) + " -> " + str(value)

    print_args = print_args + '\n' + '*'*45
    print(print_args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu    
    trainloader = get_dataset(args)

    model = rn.ResNet20(num_blocks=[3, 3, 3], num_classes=10).cuda()
    mask = torch.load("./lottery_ticket/checkpoints/CIFAR-10/{name}/mask_level{level}.pth".format(name=args.name, level=args.pruning_level))
    checkpoint = torch.load("./lottery_ticket/checkpoints/CIFAR-10/{name}/level{level}_ep100.pth".format(name=args.name, level=args.pruning_level))

    pruned_model = PrunedModel(model, mask)
    pruned_model.load_state_dict(checkpoint)
    pruned_model.eval()
    pruned_model.cuda()
    
    gradient_mean_dict = get_gradient_mean_dict(pruned_model, trainloader)

    directory = './gradients/{name}/{form}/level_{level}/'.format(name=args.name, form=args.saved_form, level=args.pruning_level)
    if not os.path.exists(directory):
        os.makedirs(directory)

    torch.save(gradient_mean_dict, directory + '{name}.pth'.format(name="mean"))
    print("\nGMM Mean saved in {directory}{name}.pth".format(directory=directory, name="mean"))

    gradient_covariance_dict = get_gradient_covariance_dict(pruned_model, trainloader, gradient_mean_dict)
    gradient_covariance_dict = fix_not_PSD_covariance(gradient_covariance_dict)

    directory = './gradients/{name}/{form}/level_{level}/'.format(name=args.name, form=args.saved_form, level=args.pruning_level)
    if not os.path.exists(directory):
        os.makedirs(directory)

    torch.save(gradient_covariance_dict, directory + '{name}.pth'.format(name="covariance"))
    print("\nGMM Covariance saved in {directory}{name}.pth".format(directory=directory, name="covariance"))
