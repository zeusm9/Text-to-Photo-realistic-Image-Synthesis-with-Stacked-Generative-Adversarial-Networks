#Text to Photo realistic Image Synthesis with Stacked Generative Adversarial Networks

Pytorch implementation of [StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks](https://arxiv.org/pdf/1612.03242v2.pdf)  by Han Zhang, Tao Xu, Hongsheng Li, Shaoting Zhang, Xiaogang Wang, Xiaolei Huang, Dimitris Metaxas. using [CUB-200 dataset](http://www.vision.caltech.edu/visipedia/CUB-200.html).

![Network Structure](https://github.com/hanzhanggit/StackGAN-Pytorch/blob/master/examples/framework.jpg)

Text to image synthesis is a computer vision task having many practical applications. In this work I used stacked Generative Adversarial Networks to generate photo-realistic images conditioned on text descriptions.

###Requirements
* Python 3.8
* Pytorch
* Numpy
* Scipy


###Data

1. Download preprocessed char-CNN-RNN text [embeddings for CUB-200](https://drive.google.com/file/d/0B3y_msrWZaXLT1BZdVdycDY5TEE/view), extract train and test folders and save them to data/birds/
2. Download [CUB-200-2011 images data](https://drive.google.com/file/d/1hbzc_P1FuxMkcabkgn9ZKinBwW683j45/view). Extract them to data/birds/

###Training

Train using stage 1 network:
```
python main.py –data_dir ../data/birds –train y –stage 1
```

Train using stage 2 network:
```
python main.py –data_dir ../data/birds –train y –stage 2
```

### Evaluate and generate samples:
```
python main.py –data_dir ../data/birds –train n
```

### References

* [StackGAN-v2-pytorch](https://github.com/hanzhanggit/StackGAN-Pytorch)
* [StackGAN-v1: Pytorch implementation](https://github.com/hanzhanggit/StackGAN-Pytorch)

