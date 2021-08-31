import os
import abc
import sys
import copy
import time
import torch
import shutil
import argparse
import numpy as np
import torchvision
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import models.resnet as rn
import torch.nn.init as init
import models.densenet as dn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
from models.purned_model import PrunedModel
import torchvision.transforms as transforms

# Specifiying and setting cli arguments
parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')
parser.add_argument('--gpu', default='0', type=str, help='which gpu to use')
parser.add_argument('--in-dataset', default="CIFAR-10", type=str, help='in-distribution dataset')
parser.add_argument('--model-arch', default='densenet_100', type=str, help='model architecture')
parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0.0001, type=float,
                    help='weight decay (default: 0.0001)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    help='print frequency (default: 50)')
parser.add_argument('--growth', default=12, type=int,
                    help='number of new channels per layer (default: 12)')
parser.add_argument('--droprate', default=0.0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--reduce', default=0.5, type=float,
                    help='compression rate in transition stage (default: 0.5)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='To not use bottleneck block')
parser.add_argument('--resume', default=None, type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', required=True, type=str,
                    help='name of experiment')
parser.add_argument('--random_seed', default=1, type=int,
                    help='The seed used for torch & numpy')
parser.add_argument('--pruning_levels', default=3, type=int,
                    help='total iterations of pruning')
parser.add_argument('--pruning_fraction', default=0.2, type=float,
                    help='fraction of network pruned')
parser.add_argument('--rewind', default=0, type=int,
                    help='number of iteration to rewind back to')
parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)

# Setting up rewind_iteration
rewind_iteration = 0

# Declaring empty rewind set of weights
rewind_state_dict = None

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print('---------------> Accuracy {top1.avg:.3f} <---------------'.format(top1=top1))
    return top1.avg


def adjust_learning_rate(optimizer, epoch, lr_schedule=[50, 75, 90]):
    """Sets the learning rate to the initial LR decayed by 10 after 40 and 80 epochs"""
    lr = args.lr
    if epoch >= lr_schedule[0]:
        lr *= 0.1
    
    if epoch >= lr_schedule[1]:
        lr *= 0.1
    
    if epoch >= lr_schedule[2]:
        lr *= 0.1
    
    # lr = args.lr * (0.1 ** (epoch // 60)) * (0.1 ** (epoch // 80))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def prune(trained_model, level, current_mask = None, pruning_fraction = 0.2):
    prunable_layer_names = [name + '.weight' for name, module in trained_model.named_modules() if
                            isinstance(module, torch.nn.modules.conv.Conv2d) or
                            isinstance(module, torch.nn.modules.linear.Linear)]
    if current_mask is None:
        mask = dict()
        for name in prunable_layer_names:
            mask[name] = torch.ones(list(trained_model.state_dict()[name].shape))

        current_mask = mask
    else:
        current_mask = {k: v.cpu().numpy() for k,v in current_mask.items()}

    if level == 0:
        return current_mask
    else:
        # Determine the number of weights that need to be pruned.
        number_of_remaining_weights = np.sum([np.sum(v) for v in current_mask.values()])
        number_of_weights_to_prune = np.ceil(pruning_fraction * number_of_remaining_weights).astype(int)

        # Determine which layers can be pruned.
        prunable_tensors = set(prunable_layer_names)

        # Get the model weights.        
        weights = {k: v.clone().cpu().detach().numpy()
                    for k, v in trained_model.state_dict().items()
                    if k in prunable_tensors}

        # Create a vector of all the unpruned weights in the model.
        weight_vector = np.concatenate([v[current_mask[k] == 1] for k, v in weights.items()])
        threshold = np.sort(np.abs(weight_vector))[number_of_weights_to_prune]

        new_mask = {k: torch.Tensor(np.where(np.abs(v) > threshold, current_mask[k], np.zeros_like(v))) for k, v in weights.items()}
        for k in current_mask:
            if k not in new_mask:
                new_mask[k] = current_mask[k]

        return new_mask


def establish_initial_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM) or isinstance(m, nn.LSTMCell) or isinstance(m, nn.GRU) or isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


def sparsity(mask):
    """Return the percent of weights that have been pruned as a decimal."""
    unpruned = torch.sum(torch.tensor([torch.sum(v) for v in mask.values()]))
    total = torch.sum(torch.tensor([torch.sum(torch.ones_like(v)) for v in mask.values()]))
    return 1 - unpruned.float() / total.float()


def save_checkpoint(model, mask, level, epoch):
    # Saving Model to disk
    directory = "./lottery_ticket/checkpoints/{in_dataset}/{name}/".format(in_dataset=args.in_dataset, name=args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + 'level{}_ep{}.pth'.format(level, epoch)
    torch.save(model.state_dict(), filename)

    # Saving Mask to disk
    filename = directory + 'mask_level{}.pth'.format(level)
    torch.save(mask, filename)

    # Printing out sparsity report
    print("Level {} with sparsity {}".format(level, sparsity(mask)))


def train(train_loader, model, criterion, optimizer, epoch):
    global rewind_iteration
    global rewind_state_dict

    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    nat_losses = AverageMeter()
    nat_top1 = AverageMeter()
    adv_losses = AverageMeter()
    adv_top1 = AverageMeter()

    # Switch to train mode
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        if rewind_iteration == args.rewind:
            # Saving rewind set of weights
            rewind_state_dict = copy.deepcopy(model.state_dict())
            directory = "./lottery_ticket/checkpoints/{in_dataset}/{name}/".format(in_dataset=args.in_dataset, name=args.name)
            filename = directory + 'rewind_state_dict.pth'
            torch.save(rewind_state_dict, filename)
            print("> Saving rewind_state_dict on iteration {ri}({index}) <".format(ri=rewind_iteration, index=i))

        input = input.cuda()
        target = target.cuda()
        nat_output = model(input)
        nat_loss = criterion(nat_output, target)

        # measure accuracy and record loss
        nat_prec1 = accuracy(nat_output.data, target, topk=(1,))[0]
        nat_losses.update(nat_loss.data, input.size(0))
        nat_top1.update(nat_prec1, input.size(0))

        # compute gradient and do SGD step
        loss = nat_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Incremeting rewind iteration counter
        rewind_iteration = rewind_iteration + 1

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    loss=nat_losses, top1=nat_top1))


def main():
    global rewind_state_dict

    # Setting up augments for training data set
    if args.augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
    else:
        transform_train = transforms.Compose([transforms.ToTensor()])
    
    # Setting up augments for testing data set
    transform_test = transforms.Compose([transforms.ToTensor()])
    kwargs = {'num_workers': 1, 'pin_memory': True}
    if args.in_dataset == "CIFAR-10":
        # Data loading code
        normalizer = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./datasets/cifar10', train=True, download=True,
                             transform=transform_train),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./datasets/cifar10', train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        lr_schedule=[50, 75, 90]
        num_classes = 10

    # Create the model
    if args.model_arch == 'densenet_100':
        model = dn.DenseNet3(depth=100, num_classes=num_classes, growth_rate=args.growth, reduction=args.reduce,
                             bottleneck=args.bottleneck, dropRate=args.droprate, normalizer=normalizer)
    elif args.model_arch == 'resnet_20':
        model = rn.ResNet20(num_blocks=[3, 3, 3], num_classes=10)
    elif args.model_arch == 'resnet_32':
        model = rn.ResNet32(num_blocks=[5, 5, 5], num_classes=10)
    else:
        assert False, 'Not supported model arch: {}'.format(args.model_arch)

    # Establishing initial set of weigths
    model.apply(establish_initial_weights)

    mask = None
    model = model.cuda()
    cudnn.benchmark = True

    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.weight_decay)

    # Run pretraining if rewind is used
    print('---------------> Starting Pretraining Process <---------------')

    while(rewind_state_dict is None):
        # Train for one epoch on the training set
        train(train_loader, model, criterion, optimizer, 0)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, 0)

    if rewind_state_dict is None:
        assert False, 'Rewind state_dict not correctly initialized'

    # Run through prunning levels
    for level in range(0, args.pruning_levels+1):
        print('---------------> Starting Pruning Level {} <---------------'.format(level))

        # Load initial set of weights set through rewind iteration
        model.load_state_dict(rewind_state_dict)

        # Prune the model (sparse global)
        mask = prune(trained_model=model, level=level, current_mask=mask)
        pruned_model = PrunedModel(model, mask).cuda()

        for epoch in range(0, args.epochs):
            adjust_learning_rate(optimizer, epoch, lr_schedule)

            # Train for one epoch on the training set
            train(train_loader, pruned_model, criterion, optimizer, epoch)

            # evaluate on validation set
            prec1 = validate(val_loader, pruned_model, criterion, epoch)

        # Save each epoch
        save_checkpoint(pruned_model, mask, level, epoch+1)


if __name__ == '__main__':
    # Grabbing cli arguments and printing results
    args = parser.parse_args()
    print_args = '*'*45
    for key,value in args._get_kwargs():
        print_args = print_args + '\n- ' + str(key) + " -> " + str(value)

    print_args = print_args + '\n' + '*'*45
    print(print_args)

    # Creating checkpoint directory
    directory = "./lottery_ticket/checkpoints/{in_dataset}/{name}/".format(in_dataset=args.in_dataset, name=args.name)
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
    np.random.seed(args.random_seed)

    # Running main training method
    main()
