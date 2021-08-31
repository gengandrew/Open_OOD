import os
import abc
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from models import vae
import models.resnet as rn
import torch.optim as optim
from sklearn.covariance import oas
from torch.nn import functional as F
import torchvision.datasets as datasets
from models.purned_model import PrunedModel
import torchvision.transforms as transforms


parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')
parser.add_argument('--gpu', default='0', type=str, help='which gpu to use')
parser.add_argument('--dataset', default="CIFAR-10", type=str, help='distribution dataset')
parser.add_argument('--saved_form', default="gmm", type=str, help='format of saved result')
parser.add_argument('-b', '--batch-size', default=1, type=int, help='batch size (default: 128)')
parser.add_argument('--epochs', default=25, type=int, help='number of total epochs to run')
parser.add_argument('--name', required=True, type=str, help='name of experiment')
parser.add_argument('--pruning_level', default=0, type=int, help='total iterations of pruning')


def get_dataset(args):
    transform = transforms.Compose([transforms.ToTensor()])
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


def get_gradient_sample_dict(pruned_model, trainloader):
    criterion = nn.CrossEntropyLoss().cuda()
    gradient_sample_dict = dict()

    for i, (inputs, targets) in enumerate(tqdm(trainloader)):
        if i > 10000:
            break

        inputs = inputs.cuda()
        class_gradient_list = []
        class_gradient_losses = []

        for class_index in range(0, 10):
            outputs = pruned_model(inputs)
            targets = torch.ones(targets.shape).long() * class_index
            targets = targets.cuda()

            pruned_model.zero_grad()
            loss = criterion(outputs, targets)
            class_loss_value = loss.item()
            loss.backward()

            model_grad_dict = extract_gradients(pruned_model)
            class_gradient_losses.append(class_loss_value)
            class_gradient_list.append(model_grad_dict)
        
        min_class = np.argmin(class_gradient_losses)

        for name, gradient in class_gradient_list[min_class].items():
            if 'bn' in name or 'bias' in name:
                continue

            if 'layer2.0.conv2.weight' in name or 'linear.weight' in name:
                flatten_gradient = torch.flatten(gradient).cpu().numpy()

                if name not in gradient_sample_dict:
                    gradient_sample_dict[name] = flatten_gradient
                else:
                    gradient_sample_dict[name] = np.vstack((gradient_sample_dict[name], flatten_gradient))
    
    return gradient_sample_dict


def train_vae(model, optimizer, train_loader, epoch):
    # Toggle the vae model to train
    model.train()
    train_loss = 0

    for index, (data) in enumerate(train_loader):
        data = data.cuda()
        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data)
        loss = model.loss_function(recon_batch, data, mu, logvar)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if index % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, index * len(data), len(train_loader.dataset),
                100. * index / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
    return (mu, logvar)


if __name__ == '__main__':
    args = parser.parse_args()
    print_args = '*'*45
    for key,value in args._get_kwargs():
        print_args = print_args + '\n- ' + str(key) + " -> " + str(value)

    print_args = print_args + '\n' + '*'*45
    print(print_args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu    
    trainloader = get_dataset(args)

    if args.name == "resnet":
        model = rn.ResNet20(num_blocks=[3, 3, 3], num_classes=10).cuda()
    elif args.name == "resnet_32":
        model = rn.ResNet32(num_blocks=[5, 5, 5], num_classes=10).cuda()
    
    mask = torch.load("./lottery_ticket/checkpoints/CIFAR-10/{name}/mask_level{level}.pth".format(name=args.name, level=args.pruning_level))
    checkpoint = torch.load("./lottery_ticket/checkpoints/CIFAR-10/{name}/level{level}_ep100.pth".format(name=args.name, level=args.pruning_level))

    pruned_model = PrunedModel(model, mask)
    pruned_model.load_state_dict(checkpoint)
    pruned_model.eval()
    pruned_model.cuda()

    gradient_sample_dict = get_gradient_sample_dict(pruned_model, trainloader)

    # directory = './gradients/{name}/{form}/level_{level}/'.format(name="test_resnet_vae", form=args.saved_form, level=args.pruning_level)
    # if not os.path.exists(directory):
    #     os.makedirs(directory)

    # torch.save(gradient_sample_dict, directory + '{name}.pth'.format(name="test_resnet_vae"))
    # print("\nGMM samples saved in {directory}{name}.pth".format(directory=directory, name="samples"))

    # SEED = 1
    # torch.manual_seed(SEED)
    # torch.cuda.manual_seed(SEED)

    kwargs = {'num_workers': 1, 'pin_memory': True}
    dataset = gradient_sample_dict["linear.weight"]
    train_loader = torch.utils.data.DataLoader(torch.Tensor(dataset), batch_size=args.batch_size, shuffle=True, **kwargs)

    model = vae.VAE(dataset.shape[1], 64).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("\nStarting variational autoencoder training")
    for epoch in range(1, args.epochs + 1):
        (mu, logvar) = train_vae(model, optimizer, train_loader, epoch)

    mu = mu[mu.shape[0]-1]
    logvar = logvar[logvar.shape[0]-1]
    std = torch.exp(0.5*logvar)
    normal_dict = {"mean": mu, "std": std}

    directory = './gradients/{name}/{form}/level_{level}/'.format(name="test_resnet_vae", form=args.saved_form, level=args.pruning_level)
    if not os.path.exists(directory):
        os.makedirs(directory)

    torch.save(model.state_dict(), directory + '{name}.pth'.format(name="vae_checkpoint"))
    print("\nVAE model dictionary saved in {directory}{name}.pth".format(directory=directory, name="vae_checkpoint"))

    torch.save(normal_dict, directory + '{name}.pth'.format(name="mean_std_checkpoint"))
    print("\nMean and std dictionary saved in {directory}{name}.pth".format(directory=directory, name="mean_std_checkpoint"))
