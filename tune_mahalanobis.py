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
import models.resnet18 as rn
from utils import metric, sample_estimator, get_Mahalanobis_score


def print_results(results, stypes):
    mtypes = ['FPR', 'DTERR', 'AUROC', 'AUIN', 'AUOUT']
    for stype in stypes:
        print(' OOD detection method: ' + stype)
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        
        print('\n{val:6.2f}'.format(val=100.*results[stype]['FPR']), end='')
        print(' {val:6.2f}'.format(val=100.*results[stype]['DTERR']), end='')
        print(' {val:6.2f}'.format(val=100.*results[stype]['AUROC']), end='')
        print(' {val:6.2f}'.format(val=100.*results[stype]['AUIN']), end='')
        print(' {val:6.2f}\n'.format(val=100.*results[stype]['AUOUT']), end='')
        print('')


def tune_mahalanobis_hyperparams(name, in_dataset, model_arch, epochs=100, batch_size=10, layers=100):
    print('Tuning mahalanobis hyper-parameters')
    stypes = ['mahalanobis']

    if in_dataset == "CIFAR-10":
        trainset= torchvision.datasets.CIFAR10('./datasets/cifar10', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        trainloaderIn = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.CIFAR10(root='./datasets/cifar10', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        testloaderIn = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)
        num_classes = 10
    else:
        assert False, 'Not supported in_dataset: {}'.format(in_dataset)

    if model_arch == 'resnet_18':
        model = model = rn.ResNet18()
    elif model_arch == 'densenet':
        model = dn.DenseNet3(layers, num_classes, normalizer=transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0)))
    else:
        assert False, 'Not supported model arch: {}'.format(model_arch)

    checkpoint = torch.load("./trained_models/{in_dataset}/{name}/epoch_{epochs}.pth".format(in_dataset=in_dataset, name=name, epochs=epochs))
    model.load_state_dict(checkpoint)
    model.eval()
    model.cuda()

    # Set information about feature extaction
    temp_x = torch.rand(2,3,32,32)
    temp_x = Variable(temp_x).cuda()
    temp_list = model.feature_list(temp_x)[1]
    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1

    print('get sample mean and covariance')
    sample_mean, precision = sample_estimator(model, num_classes, feature_list, trainloaderIn)

    print('train logistic regression model')
    m = 500
    train_in = []
    train_in_label = []
    train_out = []
    val_in = []
    val_in_label = []
    val_out = []
    cnt = 0
    for data, target in testloaderIn:
        data = data.numpy()
        target = target.numpy()
        for x, y in zip(data, target):
            cnt += 1
            if cnt <= m:
                train_in.append(x)
                train_in_label.append(y)
            elif cnt <= 2*m:
                val_in.append(x)
                val_in_label.append(y)

            if cnt == 2*m:
                break
        if cnt == 2*m:
            break

    print('In', len(train_in), len(val_in))
    criterion = nn.CrossEntropyLoss().cuda()
    adv_noise = 0.05
    for i in range(int(m/batch_size) + 1):
        if i*batch_size >= m:
            break

        data = torch.tensor(train_in[i*batch_size:min((i+1)*batch_size, m)])
        target = torch.tensor(train_in_label[i*batch_size:min((i+1)*batch_size, m)])
        data = data.cuda()
        target = target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)

        model.zero_grad()
        inputs = Variable(data.data, requires_grad=True).cuda()
        output = model(inputs)
        loss = criterion(output, target)
        loss.backward()

        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float()-0.5)*2

        adv_data = torch.add(input=inputs.data, other=gradient, alpha=adv_noise)
        adv_data = torch.clamp(adv_data, 0.0, 1.0)

        train_out.extend(adv_data.cpu().numpy())

    for i in range(int(m/batch_size) + 1):
        if i*batch_size >= m:
            break
        data = torch.tensor(val_in[i*batch_size:min((i+1)*batch_size, m)])
        target = torch.tensor(val_in_label[i*batch_size:min((i+1)*batch_size, m)])
        data = data.cuda()
        target = target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)

        model.zero_grad()
        inputs = Variable(data.data, requires_grad=True).cuda()
        output = model(inputs)
        loss = criterion(output, target)
        loss.backward()

        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float()-0.5)*2

        adv_data = torch.add(input=inputs.data, other=gradient, alpha=adv_noise)
        adv_data = torch.clamp(adv_data, 0.0, 1.0)

        val_out.extend(adv_data.cpu().numpy())

    print('Out', len(train_out),len(val_out))

    train_lr_data = []
    train_lr_label = []
    train_lr_data.extend(train_in)
    train_lr_label.extend(np.zeros(m))
    train_lr_data.extend(train_out)
    train_lr_label.extend(np.ones(m))
    train_lr_data = torch.tensor(train_lr_data)
    train_lr_label = torch.tensor(train_lr_label)
    best_fpr = 1.1
    best_magnitude = 0.0
    for magnitude in [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]:
        train_lr_Mahalanobis = []
        total = 0
        for data_index in range(int(np.floor(train_lr_data.size(0) / batch_size))):
            data = train_lr_data[total : total + batch_size].cuda()
            total += batch_size
            Mahalanobis_scores = get_Mahalanobis_score(data, model, num_classes, sample_mean, precision, num_output, magnitude)
            train_lr_Mahalanobis.extend(Mahalanobis_scores)

        train_lr_Mahalanobis = np.asarray(train_lr_Mahalanobis, dtype=np.float32)
        regressor = LogisticRegressionCV(n_jobs=-1).fit(train_lr_Mahalanobis, train_lr_label)
        print('Logistic Regressor params:', regressor.coef_, regressor.intercept_)

        confidence_scores_in = [] # confidence_mahalanobis_In
        confidence_scores_out = [] # confidence_mahalanobis_Out
        
        ########################################In-distribution###########################################
        print("Processing in-distribution images")
        count = 0
        for i in range(int(m/batch_size) + 1):
            if i * batch_size >= m:
                break

            images = torch.tensor(val_in[i * batch_size : min((i+1) * batch_size, m)]).cuda()
            batch_size = images.shape[0]
            Mahalanobis_scores = get_Mahalanobis_score(images, model, num_classes, sample_mean, precision, num_output, magnitude)
            confidence_scores= regressor.predict_proba(Mahalanobis_scores)[:, 1]
            for k in range(batch_size):
                confidence_scores_in.append(-confidence_scores[k])

            count += batch_size
            print("{:4}/{:4} images processed.".format(count, m))

        ###################################Out-of-Distributions#####################################
        print("Processing out-of-distribution images")
        count = 0
        for i in range(int(m/batch_size) + 1):
            if i * batch_size >= m:
                break

            images = torch.tensor(val_out[i * batch_size : min((i+1) * batch_size, m)]).cuda()
            batch_size = images.shape[0]

            Mahalanobis_scores = get_Mahalanobis_score(images, model, num_classes, sample_mean, precision, num_output, magnitude)

            confidence_scores= regressor.predict_proba(Mahalanobis_scores)[:, 1]

            for k in range(batch_size):
                confidence_scores_out.append(-confidence_scores[k])

            count += batch_size
            print("{:4}/{:4} images processed.".format(count, m))

        results = metric(np.array(confidence_scores_in), np.array(confidence_scores_out), stypes)
        print_results(results, stypes)
        fpr = results['mahalanobis']['FPR']
        if fpr < best_fpr:
            print("regressor updated")
            best_fpr = fpr
            best_magnitude = magnitude
            best_regressor = regressor

    print('Best Logistic Regressor params:', best_regressor.coef_, best_regressor.intercept_)
    print('Best magnitude', best_magnitude)

    return sample_mean, precision, best_regressor, best_magnitude
