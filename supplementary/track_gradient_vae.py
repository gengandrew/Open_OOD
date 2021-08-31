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
import torchvision.datasets as datasets
from models.purned_model import PrunedModel
import torchvision.transforms as transforms
from scipy.stats import multivariate_normal

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
    if  args.dataset == 'SVHN':
        testsetout = svhn.SVHN('datasets/ood_datasets/svhn/', split='test', transform=transforms.ToTensor(), download=False)
        testloader = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size, shuffle=True, num_workers=2)
    elif args.dataset == 'dtd':
        testsetout = datasets.ImageFolder(root="datasets/ood_datasets/dtd/images",
                                    transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()]))
        testloader = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size, shuffle=True, num_workers=2)
    elif args.dataset == 'iSUN' or args.dataset == 'LSUN' or args.dataset == 'LSUN_resize':
        testsetout = datasets.ImageFolder("./datasets/ood_datasets/{}".format(args.dataset),
                                    transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()]))
        testloader = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size, shuffle=True, num_workers=2)
    else:
        transform = transforms.Compose([transforms.ToTensor()])
        testset = datasets.CIFAR10(root='./datasets/cifar10', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    return testloader


def extract_gradients(pruned_model):
    model_grad_dict = dict()
    for name, param in pruned_model.model.named_parameters():
        if hasattr(pruned_model, PrunedModel.to_mask_name(name)):
            pruned_grad = param.grad * getattr(pruned_model, PrunedModel.to_mask_name(name))
            model_grad_dict[name] = pruned_grad
        else:
            model_grad_dict[name] = param.grad

    return model_grad_dict


def calculate_reconstructed_prob(mu, logvar, sample):
    std = torch.exp(0.5*logvar)
    reconstructed_prob = 0

    sample = sample.cpu().detach().numpy()[0]
    mu = mu.cpu().detach().numpy()[0]
    std = std.cpu().detach().numpy()[0]

    for l_index in range(100):
        std = std + 0.00001
        index_prob = multivariate_normal.pdf(sample, mu, np.diag(std))
        reconstructed_prob += index_prob
    
    return reconstructed_prob/100


if __name__ == '__main__':
    args = parser.parse_args()
    print_args = '*'*45
    for key,value in args._get_kwargs():
        print_args = print_args + '\n- ' + str(key) + " -> " + str(value)

    print_args = print_args + '\n' + '*'*45
    print(print_args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu    
    testloader = get_dataset(args)

    model = rn.ResNet20(num_blocks=[3, 3, 3], num_classes=10).cuda()
    mask = torch.load("./lottery_ticket/checkpoints/CIFAR-10/{name}/mask_level{level}.pth".format(name=args.name, level=args.pruning_level))
    checkpoint = torch.load("./lottery_ticket/checkpoints/CIFAR-10/{name}/level{level}_ep100.pth".format(name=args.name, level=args.pruning_level))

    pruned_model = PrunedModel(model, mask)
    pruned_model.load_state_dict(checkpoint)
    pruned_model.eval()
    pruned_model.cuda()

    vae_encoder = vae.VAE(640, 64)
    vae_checkpoint = torch.load("./gradients/test_resnet_vae/gmm/level_0/vae_checkpoint.pth")

    vae_encoder.load_state_dict(vae_checkpoint)
    vae_encoder.eval()
    vae_encoder.cuda()
    
    CE = nn.CrossEntropyLoss().cuda()
    gradient_vae_score_dict = dict()

    print("# ------------------------ Iterating through testing set ------------------------ #")
    for i, (inputs, targets) in enumerate(tqdm(testloader)):       
        inputs = inputs.cuda()
        class_gradient_list = []
        class_gradient_losses = []

        for class_index in range(0, 10):
            outputs = pruned_model(inputs)
            targets = torch.ones(targets.shape).long() * class_index
            targets = targets.cuda()

            pruned_model.zero_grad()
            loss = CE(outputs, targets)
            class_loss_value = loss.item()
            loss.backward()

            model_grad_dict = extract_gradients(pruned_model)
            class_gradient_losses.append(class_loss_value)
            class_gradient_list.append(model_grad_dict)
        
        min_class = np.argmin(class_gradient_losses)

        for name, gradient in class_gradient_list[min_class].items():
            if 'bn' in name or 'bias' in name:
                continue

            if 'linear.weight' in name:
                flatten_gradient = torch.flatten(gradient)
                vae_encoder.zero_grad()

                # encoded_sample, mu, logvar = vae_encoder(flatten_gradient)
                recon_batch, mu, logvar = vae_encoder(flatten_gradient)
                loss = vae_encoder.loss_function(recon_batch, flatten_gradient, mu, logvar)

                score = loss.item()
                # score = calculate_reconstructed_prob(mu, logvar, encoded_sample)

                if name not in gradient_vae_score_dict:
                    gradient_vae_score_dict[name] = []
                
                gradient_vae_score_dict[name].append(score)

    directory = './gradients/{name}/{form}/level_{level}/'.format(name="test_resnet_vae_loss", form=args.saved_form, level=args.pruning_level)
    if not os.path.exists(directory):
        os.makedirs(directory)

    torch.save(gradient_vae_score_dict, directory + '{dataset}_{name}.pth'.format(dataset=args.dataset, name="score"))
    print("\nVAE scores saved in {directory}{dataset}_{name}.pth".format(directory=directory, dataset=args.dataset, name="score"))