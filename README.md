# Gradients for Out of Distribution Detection

## About
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed hendrerit ligula a tempus vulputate. Phasellus augue nibh, gravida eget tempus non, fringilla nec magna. Vivamus semper, justo vitae vulputate molestie, augue mauris tempor diam.

## Getting Started

### Downloading In-distribution Dataset
* [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html): included in PyTorch.

### Downloading Auxiliary Unlabeled Dataset

To download **80 Million Tiny Images** dataset. In the **root** directory, run
```
cd datasets/unlabeled_datasets/80M_Tiny_Images
wget http://horatio.cs.nyu.edu/mit/tiny/data/tiny_images.bin
```

### Downloading Out-of-distribution Test Datasets

We provide links and instructions to download each dataset:

* [SVHN](http://ufldl.stanford.edu/housenumbers/test_32x32.mat): download it and place it in the folder of `datasets/ood_datasets/svhn`. Then run `python select_svhn_data.py` to generate test subset.
* [Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz): download it and place it in the folder of `datasets/ood_datasets/dtd`.
* [Places365](http://data.csail.mit.edu/places/places365/test_256.tar): download it and place it in the folder of `datasets/ood_datasets/places365/test_subset`. We randomly sample 10,000 images from the original test dataset. We provide the file names for the images that we sample in `datasets/ood_datasets/places365/test_subset/places365_test_list.txt`.
* [LSUN-C](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz): download it and place it in the folder of `datasets/ood_datasets/LSUN`.
* [LSUN-R](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz): download it and place it in the folder of `datasets/ood_datasets/LSUN_resize`.
* [iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz): download it and place it in the folder of `datasets/ood_datasets/iSUN`.

For example, run the following commands in the **root** directory to download **LSUN-C**:
```
cd datasets/ood_datasets
wget https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz
tar -xvzf LSUN.tar.gz
```

## Running Experiments

### Training Resnet Networks
```
python train_energy.py --name=resnet --gpu=0
python train_original.py --name=resnet_bce --gpu=0
```

### Tracking Network Gradient
```
python track_gradient.py --name=resnet --dataset=LSUN --score=min --gpu=0
python track_gradient_original.py --name=resnet_bce --dataset=LSUN --gpu=0
```

### Computing Performance Metrics
```
python compute_gradient_metrics.py --name=resnet --saved_form=norm_single --score=min --dataset=LSUN --gpu=0
python compute_gradient_metrics.py --name=resnet_bce --saved_form=original_single --dataset=LSUN --gpu=0
```

### Experimental Results
* `gradients` directory: All `track_gradient.py` results will be saved in dict form in this directory