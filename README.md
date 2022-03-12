A GAN project that turns images into pixelart

Turn videos and images to pixelart with the following commands

# Installation
run conda install requirements.yml

# Project Outline
util folder contains tools used to scrape and clean data.

Using cleaned data from source domain A and target domain B, a U-net generator is trained to transform images from A to B (Conditional GAN).

The generator is trained in a zero-sum game setting with a discriminator and siamese network, which provide gradients instructing how to generate convincing images, as well as how to organize the output of the function (the range of the function G: A -> B).

One issue with the used TravelGAN model is that it supports images of at most 256x256 res. To overcome this limitation, we set the generator to produce 128x128 squares and concatenate a 2D grid of four squares to ultimately feed the discriminator a 256x256 image.

Datasets contain images that have their widths scaled down by a factor of 2, so that square output from the generator can be scaled up to landscape images.

# Goals and Notes
The pixelart filter should work on images and also work seamlessly on video.

Will video have a good enough understanding of state?

There should be more features added as well
Use a depth map to create some part of the image a certain way (think matrix/code), and maybe control depth using music?

Maybe we could also have a custom sky?

Dataset augmentations currently include both horizontal mirror and 3 rotation augmentations