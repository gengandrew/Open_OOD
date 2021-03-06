def eval_ood_detector(base_dir, in_dataset, out_datasets, batch_size, method, method_args, name, epochs, adv, corrupt, adv_corrupt, adv_args, mode_args):

    if adv:
        in_save_dir = os.path.join(base_dir, in_dataset, method, name, 'adv', str(int(adv_args['epsilon'])))
    elif adv_corrupt:
        in_save_dir = os.path.join(base_dir, in_dataset, method, name, 'adv_corrupt', str(int(adv_args['epsilon'])))
    elif corrupt:
        in_save_dir = os.path.join(base_dir, in_dataset, method, name, 'corrupt')
    else:
        in_save_dir = os.path.join(base_dir, in_dataset, method, name, 'nat')

    if not os.path.exists(in_save_dir):
        os.makedirs(in_save_dir)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if in_dataset == "CIFAR-10":
        normalizer = transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0))
        testset = torchvision.datasets.CIFAR10(root='./datasets/cifar10', train=False, download=True, transform=transform)
        testloaderIn = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=True, num_workers=2)
        num_classes = 10
        num_reject_classes = 5
    elif in_dataset == "CIFAR-100":
        normalizer = transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0))
        testset = torchvision.datasets.CIFAR100(root='./datasets/cifar100', train=False, download=True, transform=transform)
        testloaderIn = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=True, num_workers=2)
        num_classes = 100
        num_reject_classes = 10
    elif in_dataset == "SVHN":
        normalizer = None
        testset = svhn.SVHN('datasets/svhn/', split='test',
                              transform=transforms.ToTensor(), download=False)
        testloaderIn = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=True, num_workers=2)
        num_classes = 10
        num_reject_classes = 5

    if method != "sofl":
        num_reject_classes = 0

    if method == "rowl" or method == "atom":
        num_reject_classes = 1

    method_args['num_classes'] = num_classes

    if args.model_arch == 'densenet':
        model = dn.DenseNet3(args.layers, num_classes + num_reject_classes, normalizer=normalizer)
    elif args.model_arch == 'resnet_20':
        model = rn.ResNet20(num_blocks=[3, 3, 3], num_classes=10)
    elif args.model_arch == 'wideresnet':
        model = wn.WideResNet(args.depth, num_classes + num_reject_classes, widen_factor=args.width, normalizer=normalizer)
    elif args.model_arch == 'densenet_ccu':
        model = dn.DenseNet3(args.layers, num_classes + num_reject_classes, normalizer=normalizer)
        gmm = torch.load("checkpoints/{in_dataset}/{name}/".format(in_dataset=args.in_dataset, name=args.name) + 'in_gmm.pth.tar')
        gmm.alpha = nn.Parameter(gmm.alpha)
        gmm_out = torch.load("checkpoints/{in_dataset}/{name}/".format(in_dataset=args.in_dataset, name=args.name) + 'out_gmm.pth.tar')
        gmm_out.alpha = nn.Parameter(gmm.alpha)
        whole_model = gmmlib.DoublyRobustModel(model, gmm, gmm_out, loglam = 0., dim=3072, classes=num_classes)
    elif args.model_arch == 'wideresnet_ccu':
        model = wn.WideResNet(args.depth, num_classes + num_reject_classes, widen_factor=args.width, normalizer=normalizer)
        gmm = torch.load("checkpoints/{in_dataset}/{name}/".format(in_dataset=args.in_dataset, name=args.name) + 'in_gmm.pth.tar')
        gmm.alpha = nn.Parameter(gmm.alpha)
        gmm_out = torch.load("checkpoints/{in_dataset}/{name}/".format(in_dataset=args.in_dataset, name=args.name) + 'out_gmm.pth.tar')
        gmm_out.alpha = nn.Parameter(gmm.alpha)
        whole_model = gmmlib.DoublyRobustModel(model, gmm, gmm_out, loglam = 0., dim=3072, classes=num_classes)
    else:
        assert False, 'Not supported model arch: {}'.format(args.model_arch)

    mask = torch.load("./lottery_ticket/checkpoints/{in_dataset}/{name}/mask_level{level}.pth".format(in_dataset=args.in_dataset, name=args.name, level=args.pruning_level))
    checkpoint = torch.load("./lottery_ticket/checkpoints/{in_dataset}/{name}/level{level}_ep{epochs}.pth".format(in_dataset=args.in_dataset, name=args.name, level=args.pruning_level, epochs=100))

    if args.model_arch == 'densenet_ccu' or args.model_arch == 'wideresnet_ccu':
        whole_model.load_state_dict(checkpoint)
    else:
        model = PrunedModel(model, mask)
        model.load_state_dict(checkpoint)

    model.eval()
    model.cuda()

    if method == "mahalanobis":
        temp_x = torch.rand(2,3,32,32)
        temp_x = Variable(temp_x).cuda()
        temp_list = model.feature_list(temp_x)[1]
        num_output = len(temp_list)
        method_args['num_output'] = num_output

    if adv or adv_corrupt:
        epsilon = adv_args['epsilon']
        iters = adv_args['iters']
        iter_size = adv_args['iter_size']

        if method == "msp" or method == "odin":
            attack_out = ConfidenceLinfPGDAttack(model, eps=epsilon, nb_iter=iters,
            eps_iter=args.iter_size, rand_init=True, clip_min=0., clip_max=1., num_classes = num_classes)
        elif method == "mahalanobis":
            attack_out = MahalanobisLinfPGDAttack(model, eps=args.epsilon, nb_iter=args.iters,
            eps_iter=iter_size, rand_init=True, clip_min=0., clip_max=1., num_classes = num_classes,
            sample_mean = sample_mean, precision = precision,
            num_output = num_output, regressor = regressor)
        elif method == "sofl":
            attack_out = SOFLLinfPGDAttack(model, eps=epsilon, nb_iter=iters,
            eps_iter=iter_size, rand_init=True, clip_min=0., clip_max=1.,
            num_classes = num_classes, num_reject_classes=num_reject_classes)
        elif method == "rowl":
            attack_out = OODScoreLinfPGDAttack(model, eps=epsilon, nb_iter=iters,
            eps_iter=iter_size, rand_init=True, clip_min=0., clip_max=1.,
            num_classes = num_classes)
        elif method == "atom":
            attack_out = OODScoreLinfPGDAttack(model, eps=epsilon, nb_iter=iters,
            eps_iter=iter_size, rand_init=True, clip_min=0., clip_max=1.,
            num_classes = num_classes)

    if not mode_args['out_dist_only']:
        t0 = time.time()

        f1 = open(os.path.join(in_save_dir, "in_scores.txt"), 'w')
        g1 = open(os.path.join(in_save_dir, "in_labels.txt"), 'w')

    ########################################In-distribution###########################################
        print("Processing in-distribution images")

        N = len(testloaderIn.dataset)
        count = 0
        for j, data in enumerate(testloaderIn):
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            curr_batch_size = images.shape[0]

            inputs = images

            scores = get_score(inputs, model, method, method_args)

            for score in scores:
                f1.write("{}\n".format(score))

            if method == "rowl":
                outputs = F.softmax(model(inputs), dim=1)
                outputs = outputs.detach().cpu().numpy()
                preds = np.argmax(outputs, axis=1)
                confs = np.max(outputs, axis=1)
            else:
                outputs = F.softmax(model(inputs)[:, :num_classes], dim=1)
                outputs = outputs.detach().cpu().numpy()
                preds = np.argmax(outputs, axis=1)
                confs = np.max(outputs, axis=1)

            for k in range(preds.shape[0]):
                g1.write("{} {} {}\n".format(labels[k], preds[k], confs[k]))

            count += curr_batch_size
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, N, time.time()-t0))
            t0 = time.time()

        f1.close()
        g1.close()

    if mode_args['in_dist_only']:
        return

    for out_dataset in out_datasets:

        out_save_dir = os.path.join(in_save_dir, out_dataset)

        if not os.path.exists(out_save_dir):
            os.makedirs(out_save_dir)

        f2 = open(os.path.join(out_save_dir, "out_scores.txt"), 'w')

        if not os.path.exists(out_save_dir):
            os.makedirs(out_save_dir)

        if out_dataset == 'SVHN':
            testsetout = svhn.SVHN('datasets/ood_datasets/svhn/', split='test',
                                  transform=transforms.ToTensor(), download=False)
            testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=batch_size,
                                             shuffle=True, num_workers=2)
        elif out_dataset == 'dtd':
            testsetout = torchvision.datasets.ImageFolder(root="datasets/ood_datasets/dtd/images",
                                        transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()]))
            testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=True,
                                                     num_workers=2)
        elif out_dataset == 'places365':
            testsetout = torchvision.datasets.ImageFolder(root="datasets/ood_datasets/places365/test_subset",
                                        transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()]))
            testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=True,
                                                     num_workers=2)
        else:
            testsetout = torchvision.datasets.ImageFolder("./datasets/ood_datasets/{}".format(out_dataset),
                                        transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()]))
            testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=batch_size,
                                             shuffle=True, num_workers=2)

    ###################################Out-of-Distributions#####################################
        t0 = time.time()
        print("Processing out-of-distribution images")

        N = len(testloaderOut.dataset)
        count = 0
        for j, data in enumerate(testloaderOut):

            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            curr_batch_size = images.shape[0]

            if adv:
                inputs = attack_out.perturb(images)
            elif corrupt:
                inputs = corrupt_attack(images, model, method, method_args, False, adv_args['severity_level'])
            elif adv_corrupt:
                corrupted_images = corrupt_attack(images, model, method, method_args, False, adv_args['severity_level'])
                inputs = attack_out.perturb(corrupted_images)
            else:
                inputs = images

            scores = get_score(inputs, model, method, method_args)

            for score in scores:
                f2.write("{}\n".format(score))

            count += curr_batch_size
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, N, time.time()-t0))
            t0 = time.time()

        f2.close()

    return


if __name__ == '__main__':
    method_args = dict()
    mode_args = dict()

    # mode_args['in_dist_only'] = args.in_dist_only
    # mode_args['out_dist_only'] = args.out_dist_only

    out_datasets = ['LSUN', 'LSUN_resize', 'iSUN', 'dtd']

    if args.method == 'msp':
        eval_ood_detector(args.base_dir, args.in_dataset, out_datasets, args.batch_size, args.method, method_args, args.name, args.epochs, args.adv, args.corrupt, args.adv_corrupt, adv_args, mode_args)
    elif args.method == "odin":
        method_args['temperature'] = 1000.0
        if args.model_arch == 'densenet':
            if args.in_dataset == "CIFAR-10":
                method_args['magnitude'] = 0.0016
            elif args.in_dataset == "CIFAR-100":
                method_args['magnitude'] = 0.0012
            elif args.in_dataset == "SVHN":
                method_args['magnitude'] = 0.0006
        elif args.model_arch == 'wideresnet':
            if args.in_dataset == "CIFAR-10":
                method_args['magnitude'] = 0.0006
            elif args.in_dataset == "CIFAR-100":
                method_args['magnitude'] = 0.0012
            elif args.in_dataset == "SVHN":
                method_args['magnitude'] = 0.0002
        elif args.model_arch == 'resnet_20':
            if args.in_dataset == "CIFAR-10":
                method_args['magnitude'] = 0.0006
            elif args.in_dataset == "CIFAR-100":
                method_args['magnitude'] = 0.0012
            elif args.in_dataset == "SVHN":
                method_args['magnitude'] = 0.0002
        else:
            assert False, 'Not supported model arch'

        eval_ood_detector(args.base_dir, args.in_dataset, out_datasets, args.batch_size, args.method, method_args, args.name, args.epochs, args.adv, args.corrupt, args.adv_corrupt, adv_args, mode_args)
    elif args.method == 'mahalanobis':
        sample_mean, precision, lr_weights, lr_bias, magnitude = np.load(os.path.join('output/mahalanobis_hyperparams/', args.in_dataset, args.name, 'results.npy'), allow_pickle=True)
        regressor = LogisticRegressionCV(cv=2).fit([[0,0,0,0],[0,0,0,0],[1,1,1,1],[1,1,1,1]], [0,0,1,1])
        regressor.coef_ = lr_weights
        regressor.intercept_ = lr_bias

        method_args['sample_mean'] = sample_mean
        method_args['precision'] = precision
        method_args['magnitude'] = magnitude
        method_args['regressor'] = regressor

        eval_ood_detector(args.base_dir, args.in_dataset, out_datasets, args.batch_size, args.method, method_args, args.name, args.epochs, args.adv, args.corrupt, args.adv_corrupt, adv_args, mode_args)
    elif args.method == 'sofl':
        eval_ood_detector(args.base_dir, args.in_dataset, out_datasets, args.batch_size, args.method, method_args, args.name, args.epochs, args.adv, args.corrupt, args.adv_corrupt, adv_args, mode_args)
    elif args.method == 'rowl':
        eval_ood_detector(args.base_dir, args.in_dataset, out_datasets, args.batch_size, args.method, method_args, args.name, args.epochs, args.adv, args.corrupt, args.adv_corrupt, adv_args, mode_args)
    elif args.method == 'atom':
        eval_ood_detector(args.base_dir, args.in_dataset, out_datasets, args.batch_size, args.method, method_args, args.name, args.epochs, args.adv, args.corrupt, args.adv_corrupt, adv_args, mode_args)
