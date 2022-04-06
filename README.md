# ADH: Composite Vision

Applied Data Hackathon 2022.

Composite Vision team project on segmenting defects in CT scans of composite material parts.

# Problem

When manufacturing [Carbon fiber reinforced polymers](https://en.wikipedia.org/wiki/Carbon-fiber-reinforced_polymers),
structural defects such as delaminations can occur inside material.
These defects can impair composite physical properties and even render a detail unacceptably defective.
Non-destructing testing is a procedure that can be used to identify these defects by means of computed tomography imaging,
ultrasound imaging or other techniques.

# Goal

In this project, we are working on a computer vision model capable of segmenting defects in CT images of a CFRP composite.

# Data

The data consists of CT images of several composite parts:

* 200 sample grayscale PNG images with annotations in VOC polygon format.
  [Link](https://disk.yandex.ru/d/9WPI8wIcv91VtA) to Yandex disk (24 Mb)

Image:
![](images/sample.jpg)

Annotations:
![](images/annotated.jpg)

Segmentation map:
![](images/segmentation.png)

U-Net model:
![](images/UNet.png)



# Model

Modified U-Net model which we proposed 



# Tools and resources

For image labeling we'll be using [LabelMe](https://pypi.org/project/labelme/).

For model prototyping -- Pytorch and torchvision.
