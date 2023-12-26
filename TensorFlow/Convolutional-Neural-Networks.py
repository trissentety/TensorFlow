"""Deep Computer Vision
In this guide we will learn how to peform image classification and object detection/recognition using deep computer vision with something called a convolutional neural network.


Exmaple
-Self Driving Cars from Tesla uses a complicated TensorFlow deep learning model
-Medicine Field
-Sports like goal line technology and detecting players on field
-Our purpose is to perform classification but can be used for object detection and recognition & facial detection recognition


The goal of our convolutional neural networks will be to classify and detect images or specific objects from within the image. We will be using image data as our features and a label for those images as our label or output.

We already know how neural networks work so we can skip through the basics and move right into explaining the following concepts.

Image Data
Convolutional Layer
Pooling Layer
CNN Architectures, pre trained models have been developed by companies such as Goggle and TensorFlow to perform classification tasks for us.
The major differences we are about to see in these types of neural networks are the layers that make them up.

Subject could very difficult to understand

Image Data
So far, we have dealt with pretty straight forward data that has 1 or 2 dimensions. Now we are about to deal with image data that is usually made up of 3 dimensions. These 3 dimensions are as follows:

Where user sees a 2 dimensional image a computer sees a 3 dimensional image
3 values for each pixel so 3 layers to represent this thought ot as stack of layers, colors or pixels telling value for each pixel
drawing 2 dimensional image with this based on width and height

Previously it used specific features to figure stuff out and flipping image could make it not classify if image is cat
Dense network sees things globally and looks at entire image to learn patterns of specific areas. Things need to be centered and very similar to use dense network to perform classification because it can't learn local patterns and apply those to different areas of image.

image height
image width
color channels
The only item in the list above you may not understand is color channels. The number of color channels represents the depth of an image and coorelates to the colors used in it. For example, an image with three channels is likely made up of rgb (red, green, blue) pixels. So, for each pixel we have three numeric values in the range 0-255 that define its color. For an image of color depth 1 we would likely have a greyscale image with one value defining each pixel, again in the range of 0-255.

Image
http://xrds.acm.org/blog/wp-content/uploads/2016/06/Figure1.png
Keep this in mind as we discuss how our network works and the input/output of each layer.


Convolutional Neural Network
Note: I will use the term convnet and convolutional neural network interchangably.

Each convolutional neural network is made up of one or many convolutional layers. These layers are different than the dense layers we have seen previously. Their goal is to find patterns from within images that can be used to classify the image or parts of it. But this may sound familiar to what our densly connected neural network in the previous section was doing, well that's becasue it is.

The fundemental difference between a dense layer and a convolutional layer is that dense layers detect patterns globally while convolutional layers detect patterns locally. When we have a densly connected layer each node in that layer sees all the data from the previous layer. This means that this layer is looking at all the information and is only capable of analyzing the data in a global capacity. Our convolutional layer however will not be densly connected, this means it can detect local patterns using part of the input data to that layer.

Let's have a look at how a densly connected layer would look at an image vs how a convolutional layer would.

This is our image; the goal of our network will be to determine whether this image is a cat or not.

Image
https://img.webmd.com/dtmcms/live/webmd/consumer_assets/site_images/article_thumbnails/reference_guide/cat_weight_ref_guide/1800x1200_cat_weight_ref_guide.jpg

Dense Layer: A dense layer will consider the ENTIRE image. It will look at all the pixels and use that information to generate some output.

Convolutional Layer: The convolutional layer will look at specific parts of the image. In this example let's say it analyzes the highlighted parts below and detects patterns there.

Patterns to look for on cat image (ears, eyes, nose, paws) flipping image would not know it is a cat using dense neural network.
Convolustional Network will learn local patterns. Instead of knowing nose exists in specific location it learns what nose looks like and can find it anywhere in image.

Convolutional network scans image and picks up features that exist in image and will pass that to a dense neural network or dense classifier and look at presence of features to determine combination of thesee presence of features that make up specific classes or objects.

Dense Neural Networks work on a global scale or pattern that are found in specific areas.

Image 
https://drive.google.com/uc?export=view&id=1M7v7S-b-zisFLI_G4ZY_RdUJQrGpJ3zt

Can you see why this might make these networks more useful?

How They Work
A dense neural network learns patterns that are present in one specific area of an image. This means if a pattern that the network knows is present in a different area of the image it will have to learn the pattern again in that new area to be able to detect it.

Let's use an example to better illustrate this.

We'll consider that we have a dense neural network that has learned what an eye looks like from a sample of dog images.

regular neural network looks at this dog image and finds 2 existing eyes where eyes are. when flipped in next image it can't find that image is dog anymore.

Dense neural network's output is bunch of numeric values,
Convolutional layers are doing is outputting feature map

Image
https://drive.google.com/uc?export=view&id=16FJKkVS_lZToQOCOOy6ohUpspWgtoQ-c

Let's say it's determined that an image is likely to be a dog if an eye is present in the boxed off locations of the image above.

Now let's flip the image.

Image
https://drive.google.com/uc?export=view&id=1V7Dh7BiaOvMq5Pm_jzpQfJTZcpPNmN0W

Since our densly connected network has only recognized patterns globally it will look where it thinks the eyes should be present. Clearly it does not find them there and therefore would likely determine this image is not a dog. Even though the pattern of the eyes is present, it's just in a different location.

Since convolutional layers learn and detect patterns from different areas of the image, they don't have problems with the example we just illustrated. They know what an eye looks like and by analyzing different parts of the image can find where it is present.

Multiple Convolutional Layers
In our models it is quite common to have more than one convolutional layer. Even the basic example we will use in this guide will be made up of 3 convolutional layers. These layers work together by increasing complexity and abstraction at each subsequent layer. The first layer might be responsible for picking up edges and short lines, while the second layer will take as input these lines and start forming shapes or polygons. Finally, the last layer might take these shapes and determine which combiantions make up a specific image.

Feature Maps
You may see me use the term feature map throughout this tutorial. This term simply stands for a 3D tensor with two spacial axes (width and height) and one depth axis. Our convolutional layers take feature maps as their input and return a new feature map that reprsents the prescence of spcific filters from the previous feature map. These are what we call response maps.

Layer Parameters
A convolutional layer is defined by two key parameters.

Filters
A filter is a m x n pattern of pixels that we are looking for in an image. The number of filters in a convolutional layer reprsents how many patterns each layer is looking for and what the depth of our response map will be. If we are looking for 32 different patterns/filters than our output feature map (aka the response map) will have a depth of 32. Each one of the 32 layers of depth will be a matrix of some size containing values indicating if the filter was present at that location or not.

Here's a great illustration from the book "Deep Learning with Python" by Francois Chollet (pg 124).

When filter is alled over image, image gets sampled at all different areas and reate an output feature map that quantifies presense of filter's pattern at different locations. Many different filters are run over image at a time. So all different feature maps telling about presence of all these features.

1 Convolusional Layer will start by doing very small simple filters suh as straight lines
Other convolusional layers on to of that will take map in created from previous layer and see map is representing example diagonal lines and next looks for curves or edges

Advantage of filter is it is slid across entire image is it looks in all different locations.
0
Image
https://drive.google.com/uc?export=view&id=1HcLvvLKvLCCGuGZPMvKYz437FbbCC2eB

Sample Size
This isn't really the best term to describe this, but each convolutional layer is going to examine n x m blocks of pixels in each image. Typically, we'll consider 3x3 or 5x5 blocks. In the example above we use a 3x3 "sample size". This size will be the same as the size of our filter.

Our layers work by sliding these filters of n x m pixels over every possible position in our image and populating a new feature map/response map indicating whether the filter is present at each location.



Borders and Padding
The more mathematical of you may have realized that if we slide a filter of let's say size 3x3 over our image well consider less positions for our filter than pixels in our input. Look at the example below.

Image from "Deep Learning with Python" by Francois Chollet (pg 126).
https://drive.google.com/uc?export=view&id=1OEfXrV16NBjwAafgBfYYcWOyBCHqaZ5M


This means our response map will have a slightly smaller width and height than our original image. This is fine but sometimes we want our response map to have the same dimensions. We can accomplish this by using something called padding.

Padding is simply the addition of the appropriate number of rows and/or columns to your input data such that each pixel can be centered by the filter.

Strides
In the previous sections we assumed that the filters would be slid continously through the image such that it covered every possible position. This is common but sometimes we introduce the idea of a stride to our convolutional layer. The stride size reprsents how many rows/cols we will move the filter each time. These are not used very frequently so we'll move on.

Pooling
You may recall that our convnets are made up of a stack of convolution and pooling layers.

The idea behind a pooling layer is to downsample our feature maps and reduce their dimensions. They work in a similar way to convolutional layers where they extract windows from the feature map and return a response map of the max, min or average values of each channel. Pooling is usually done using windows of size 2x2 and a stride of 2. This will reduce the size of the feature map by a factor of two and return a response map that is 2x smaller.

A More Detailed Look
Please refer to the video to learn how all of this happens at the lower level!

Creating a Convnet
Now it is time to create our first convnet! This example is for the purpose of getting familiar with CNN architectures, we will talk about how to improves its performance later.

This tutorial is based on the following guide from the TensorFlow documentation: https://www.tensorflow.org/tutorials/images/cnn

Dataset
The problem we will consider here is classifying 10 different everyday objects. The dataset we will use is built into tensorflow and called the CIFAR Image Dataset. It contains 60,000 32x32 color images with 6000 images of each class.

The labels in this dataset are the following:

Airplane
Automobile
Bird
Cat
Deer
Dog
Frog
Horse
Ship
Truck
We'll load the dataset and have a look at some of the images below.

