# Stroke-based Character Reconstruction

---> https://arxiv.org/abs/1806.08990 

Please refer to our extended work，https://github.com/megvii-research/ICCV2019-LearningToPaint

## Abstract

Character reconstruction for noisy character images or character images from real scene is still a challenging problem, due to the bewildering backgrounds, uneven illumination, low resolution and different distortions. We propose a stroke-based character reconstruction(SCR) method that use a weighted quadratic Bezier curve(WQBC) to represent strokes of a character. Only training on our synthetic data, our stroke extractor can achieve excellent reconstruction effect in real scenes. Meanwhile. It can also help achieve great ability in defending adversarial attacks of character recognizers.  

## Installation
Use [anaconda](https://conda.io/miniconda.html) to manage environment

```
$ conda create -n py36 python=3.6
$ source activate py36
```

### Dependencies
* [PyTorch](http://pytorch.org/) 0.4 
* [tensorboardX](https://github.com/lanpa/tensorboard-pytorch/tree/master/tensorboardX)
* [opencv-python](https://pypi.org/project/opencv-python/)

```
## Reference

[pytorch-cifar](https://github.com/kuangliu/pytorch-cifar) (model)

[FeatureSqueezing](https://github.com/uvasrg/FeatureSqueezing) (adversarial experiment)
