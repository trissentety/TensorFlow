"""
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


%tensorflow_version 2.x  # this line is not required unless you are in a notebook
import tensorflow as tf

from tensorflow.keras import datasets, layers, models #using tensorflow's data set
import matplotlib.pyplot as plt
Run
Colab only includes TensorFlow 2.x; %tensorflow_version has no effect.

#  LOAD AND SPLIT DATASET
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data() #loads in tensorfloat data setobject, download's this data set

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', #0 is airplane
               'dog', 'frog', 'horse', 'ship', 'truck']
Run
Downloading data from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
170498071/170498071 [==============================] - 5s 0us/step

               
# Let's look at a one image
IMG_INDEX = 7  # change this to look at other images

plt.imshow(train_images[IMG_INDEX] ,cmap=plt.cm.binary)
plt.xlabel(class_names[train_labels[IMG_INDEX][0]])
plt.show()
Run
Shows low resolution horse image


CNN Architecture
A common architecture for a CNN is a stack of Conv2D and MaxPooling2D layers followed by a few denesly connected layers. To idea is that the stack of convolutional and maxPooling layers extract the features from the image. Then these features are flattened and fed to densly connected layers that determine the class of an image based on the presence of features.

We will start by building the Convolutional Base.

model = models.Sequential() 
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))) # (filtersAmount, (samplesize,_), activation after applying dot product operation applies rectify linear unit put in output feature map, input_shape is what to expect in first layer)
model.add(layers.MaxPooling2D((2, 2))) #figures out size based on previous layer, 2x2 sample size stride of 2
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))


Layer 1

The input shape of our data will be 32, 32, 3 and we will process 32 filters of size 3x3 over our input data. We will also apply the activation function relu to the output of each convolution operation.

Layer 2

This layer will perform the max pooling operation using 2x2 samples and a stride of 2.

Other Layers

The next set of layers do very similar things but take as input the feature map from the previous layer. They also increase the frequency of filters from 32 to 64. We can do this as our data shrinks in spacial dimensions as it passed through the layers, meaning we can afford (computationally) to add more depth.


model.summary()  # let's have a look at our model so far
Run
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 30, 30, 32)        896       #sampling of padding so 30, 30 which is 2 pixels less
                                                                 
 max_pooling2d (MaxPooling2  (None, 15, 15, 32)        0         #srhunk shape by factor of 2
 D)                                                              
                                                                 
 conv2d_1 (Conv2D)           (None, 13, 13, 64)        18496     #Convolusion on layer above, 64 filters this time and 13, 13
                                                                 
 max_pooling2d_1 (MaxPoolin  (None, 6, 6, 64)          0         #divide above layer by factor of 2 and is rounded
 g2D)                                                            
                                                                 
 conv2d_2 (Conv2D)           (None, 4, 4, 64)          36928     #taking away values because above
                                                                 
=================================================================
Total params: 56320 (220.00 KB)
Trainable params: 56320 (220.00 KB)
Non-trainable params: 0 (0.00 Byte)
____________________________________________

Not the end of convolusional network and only tells about presence of certain features

Next is to pass information into a dence layer classifier which takes this pixel data that is calculated so extraction of features found in image and tell which combo of features map to which of the 10 classes.

After looking at the summary you should notice that the depth of our image increases but the spacial dimensions reduce drastically.



Adding Dense Layers
So far, we have just completed the convolutional base. Now we need to take these extracted features and add a way to classify them. This is why we add the following layers to our model.


model.add(layers.Flatten()) #put 4x4x64 into a straight 1 dimensional line 
model.add(layers.Dense(64, activation='relu')) #64 neuron dense layer that connect all things to it with rectify linear unit
model.add(layers.Dense(10)) #amount of classes

model.summary()
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 30, 30, 32)        896       
                                                                 
 max_pooling2d (MaxPooling2  (None, 15, 15, 32)        0         
 D)                                                              
                                                                 
 conv2d_1 (Conv2D)           (None, 13, 13, 64)        18496     
                                                                 
 max_pooling2d_1 (MaxPoolin  (None, 6, 6, 64)          0         
 g2D)                                                            
                                                                 
 conv2d_2 (Conv2D)           (None, 4, 4, 64)          36928     Convolutional base < ^ ^ ^ ^
                                                                 
 flatten (Flatten)           (None, 1024)              0         #4*4*64 = 1024 Classifier < v v both of these go together to get the class
                                                                 
 dense (Dense)               (None, 64)                65600     
                                                                 
 dense_1 (Dense)             (None, 10)                650       #10 nerons out or list of values and determine which class is predicted
                                                                 
=================================================================
Total params: 122570 (478.79 KB)
Trainable params: 122570 (478.79 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________




We can see that the flatten layer changes the shape of our data so that we can feed it to the 64-node dense layer, follwed by the final output layer of 10 neurons (one for each class).

Training
Now we will train and compile the model using the recommended hyper paramaters from tensorflow.

Note: This will take much longer than previous models!


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), #computes cross entropy loss between labels and predictions. use these basics
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=4, #access statistics from this, training data to test images, labels, train images and labels
                    validation_data=(test_images, test_labels)) #validation data
Run
Epoch 1/4
1563/1563 [==============================] - 81s 51ms/step - loss: 1.5169 - accuracy: 0.4464 - val_loss: 1.2538 - val_accuracy: 0.5495
Epoch 2/4
1563/1563 [==============================] - 74s 47ms/step - loss: 1.1530 - accuracy: 0.5917 - val_loss: 1.1137 - val_accuracy: 0.6046
Epoch 3/4
1563/1563 [==============================] - 76s 49ms/step - loss: 1.0121 - accuracy: 0.6435 - val_loss: 1.0212 - val_accuracy: 0.6398
Epoch 4/4
1563/1563 [==============================] - 78s 50ms/step - loss: 0.9201 - accuracy: 0.6767 - val_loss: 0.9854 - val_accuracy: 0.6528
Recommended 10 epochs
                    

Evaluating the Model
We can determine how well the model performed by looking at it's performance on the test data set.

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2) #model already made
print(test_acc)
Run
313/313 - 4s - loss: 0.9854 - accuracy: 0.6528 - 4s/epoch - 11ms/step
0.6528000235557556
same accuracy as above


You should be getting an accuracy of about 70%. This isn't bad for a simple model like this, but we'll dive into some better approaches for computer vision below.

Working with Small Datasets
In the situation where you don't have millions of images it is difficult to train a CNN from scratch that performs very well. This is why we will learn about a few techniques we can use to train CNN's on small datasets of just a few thousand images.

Data Augmentation
To avoid overfitting and create a larger dataset from a smaller one we can use a technique called data augmentation. This is simply performing random transofrmations on our images so that our model can generalize better. These transformations can be things like compressions, rotations, stretches and even color changes.

Fortunately, keras can help us do this. Look at the code below to an example of data augmentation.

#one image can be turned into several different images and train and pass these images to model.
Essentially rotate, flip, stretch, compress, shift, zoom it etc and pass to model should be better at generalizing because of seeing same image
Modify and augment it multiples so data set of 10,000 images into 40,000 images by doing 4 augmentations of every single image
This script does this

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

# creates a data generator object that transforms images
datagen = ImageDataGenerator( #allows specifying parameters of how to modify image
rotation_range=40,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode='nearest')

# pick an image to transform
test_img = train_images[20]
img = image.img_to_array(test_img)  # convert image to numpy arry from tensorflow data set
img = img.reshape((1,) + img.shape)  # reshape image 

i = 0

for batch in datagen.flow(img, save_prefix='test', save_format='jpeg'):  # this loops runs forever doing test 1 test 2 etc until we break, saving images to current directory with specified prefix
    plt.figure(i) #show image
    plot = plt.imshow(image.img_to_array(batch[0]))
    i += 1
    if i > 4:  # show 4 images
        break

plt.show()
Run
Shows image in various ways 5 times flipped etc in different spots


If there's still not enough images in data set

First layers work really well and only need to modify last few layers

Pretrained Models
You would have noticed that the model above takes a few minutes to train in the NoteBook and only gives an accuaracy of ~70%. This is okay but surely there is a way to improve on this.

In this section we will talk about using a pretrained CNN as apart of our own custom network to improve the accuracy of our model. We know that CNN's alone (with no dense layers) don't do anything other than map the presence of features from our input. This means we can use a pretrained CNN, one trained on millions of images, as the start of our model. This will allow us to have a very good convolutional base before adding our own dense layered classifier at the end. In fact, by using this techique we can train a very good classifier for a realtively small dataset (< 10,000 images). This is because the convnet already has a very good idea of what features to look for in an image and can find them very effectively. So, if we can determine the presence of features all the rest of the model needs to do is determine which combination of features makes a specific image.



Fine Tuning
When we employ the technique defined above, we will often want to tweak the final layers in our convolutional base to work better for our specific problem. This involves not touching or retraining the earlier layers in our convolutional base but only adjusting the final few. We do this because the first layers in our base are very good at extracting low level features lile lines and edges, things that are similar for any kind of image. Where the later layers are better at picking up very specific features like shapes or even eyes. If we adjust the final layers than we can look for only features relevant to our very specific problem.



Using a Pretrained Model
In this section we will combine the tecniques we learned above and use a pretrained model and fine tuning to classify images of dogs and cats using a small dataset.

This tutorial is based on the following guide from the TensorFlow documentation: https://www.tensorflow.org/tutorials/images/transfer_learning



#Imports
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
keras = tf.keras


Dataset
We will load the cats_vs_dogs dataset from the modoule tensorflow_datatsets.

This dataset contains (image, label) pairs where images have different dimensions and 3 color channels.


import tensorflow_datasets as tfds
tfds.disable_progress_bar()

#classify dogs versus cats with above 90% accuracy
# split the data manually into 80% training, 10% testing, 10% validation
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)
Run
Downloading and preparing dataset 786.67 MiB (download: 786.67 MiB, generated: 1.04 GiB, total: 1.81 GiB) to /root/tensorflow_datasets/cats_vs_dogs/4.0.1...
WARNING:absl:1738 images were corrupted and were skipped
Dataset cats_vs_dogs downloaded and prepared to /root/tensorflow_datasets/cats_vs_dogs/4.0.1. Subsequent calls will reuse this data.



get_label_name = metadata.features['label'].int2str  # creates a function object that we can use to get labels

# display 2 images from the dataset
for image, label in raw_train.take(5):
  plt.figure()
  plt.imshow(image)
  plt.title(get_label_name(label))
Run
Shows 5 dog and cat images with label and side measuring tool


Data Preprocessing
Since the sizes of our images are all different, we need to convert them all to the same size. We can create a function that will do that for us below.


IMG_SIZE = 160 # All images will be resized to 160x160

def format_example(image, label):
  """
  returns an image that is reshaped to IMG_SIZE 160x160 better to make image smaller than bigger it loses detail and is overall better to go smaller
  """
  image = tf.cast(image, tf.float32) #convert every pixel in image to be float 32 because it could be integers
  image = (image/127.5) - 1 #half of 255
  image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE)) #160x160
  return image, label


Now we can apply this function to all our images using .map().


train = raw_train.map(format_example) #takes every example in raw_train and applies function so resize to 160x160
validation = raw_validation.map(format_example) #also for validation test
test = raw_test.map(format_example)


Let's have a look at our images now.


for image, label in train.take(2):
  plt.figure()
  plt.imshow(image)
  plt.title(get_label_name(label))
Run
Shows 2 resized dog images with measuring tool and label

Finally we will shuffle and batch the images.


BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)


Now if we look at the shape of an original image vs the new image we will see it has been changed.


for img, label in raw_train.take(2):
  print("Original shape:", img.shape)

for img, label in train.take(2):
  print("New shape:", img.shape)
Run  
#Original shape: (262, 350, 3)
#Original shape: (409, 336, 3)
#New shape: (160, 160, 3) #reshaped values, 3 is color channel of image
#New shape: (160, 160, 3)

Picking a Pretrained Model
The model we are going to use as the convolutional base for our model is the MobileNet V2 developed at Google. This model is trained on 1.4 million images and has 1000 different classes.

We want to use this model but only its convolutional base. So, when we load in the model, we'll specify that we don't want to load the top (classification) layer. We'll tell the model what input shape to expect and to use the predetermined weights from imagenet (Googles dataset).


IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, #architecture of model, load in
                                               include_top=False, #include classifier that comes with network or not. normally trains for 1000 different classes but in this case for dogs and cats
                                               weights='imagenet') #specific save of weights, data for architecture 
Run
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160_no_top.h5
9406464/9406464 [==============================] - 1s 0us/step

                                               
                                               
base_model.summary()
Run
Model: "mobilenetv2_1.00_160"
__________________________________________________________________________________________________
 Layer (type)                Output Shape                 Param #   Connected to                  
==================================================================================================
 input_1 (InputLayer)        [(None, 160, 160, 3)]        0         []                            
                                                                                                  
 Conv1 (Conv2D)              (None, 80, 80, 32)           864       ['input_1[0][0]']             
                                                                                                  
 bn_Conv1 (BatchNormalizati  (None, 80, 80, 32)           128       ['Conv1[0][0]']               
 on)                                                                                              
                                                                                                  
 Conv1_relu (ReLU)           (None, 80, 80, 32)           0         ['bn_Conv1[0][0]']            
                                                                                                  
 expanded_conv_depthwise (D  (None, 80, 80, 32)           288       ['Conv1_relu[0][0]']          
 epthwiseConv2D)                                                                                  
                                                                                                  
 expanded_conv_depthwise_BN  (None, 80, 80, 32)           128       ['expanded_conv_depthwise[0][0
  (BatchNormalization)                                              ]']                           
                                                                                                  
 expanded_conv_depthwise_re  (None, 80, 80, 32)           0         ['expanded_conv_depthwise_BN[0
 lu (ReLU)                                                          ][0]']                        
                                                                                                  
 expanded_conv_project (Con  (None, 80, 80, 16)           512       ['expanded_conv_depthwise_relu
 v2D)                                                               [0][0]']                      
                                                                                                  
 expanded_conv_project_BN (  (None, 80, 80, 16)           64        ['expanded_conv_project[0][0]'
 BatchNormalization)                                                ]                             
                                                                                                  
 block_1_expand (Conv2D)     (None, 80, 80, 96)           1536      ['expanded_conv_project_BN[0][
                                                                    0]']                          
                                                                                                  
 block_1_expand_BN (BatchNo  (None, 80, 80, 96)           384       ['block_1_expand[0][0]']      
 rmalization)                                                                                     
                                                                                                  
 block_1_expand_relu (ReLU)  (None, 80, 80, 96)           0         ['block_1_expand_BN[0][0]']   
                                                                                                  
 block_1_pad (ZeroPadding2D  (None, 81, 81, 96)           0         ['block_1_expand_relu[0][0]'] 
 )                                                                                                
                                                                                                  
 block_1_depthwise (Depthwi  (None, 40, 40, 96)           864       ['block_1_pad[0][0]']         
 seConv2D)                                                                                        
                                                                                                  
 block_1_depthwise_BN (Batc  (None, 40, 40, 96)           384       ['block_1_depthwise[0][0]']   
 hNormalization)                                                                                  
                                                                                                  
 block_1_depthwise_relu (Re  (None, 40, 40, 96)           0         ['block_1_depthwise_BN[0][0]']
 LU)                                                                                              
                                                                                                  
 block_1_project (Conv2D)    (None, 40, 40, 24)           2304      ['block_1_depthwise_relu[0][0]
                                                                    ']                            
                                                                                                  
 block_1_project_BN (BatchN  (None, 40, 40, 24)           96        ['block_1_project[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_2_expand (Conv2D)     (None, 40, 40, 144)          3456      ['block_1_project_BN[0][0]']  
                                                                                                  
 block_2_expand_BN (BatchNo  (None, 40, 40, 144)          576       ['block_2_expand[0][0]']      
 rmalization)                                                                                     
                                                                                                  
 block_2_expand_relu (ReLU)  (None, 40, 40, 144)          0         ['block_2_expand_BN[0][0]']   
                                                                                                  
 block_2_depthwise (Depthwi  (None, 40, 40, 144)          1296      ['block_2_expand_relu[0][0]'] 
 seConv2D)                                                                                        
                                                                                                  
 block_2_depthwise_BN (Batc  (None, 40, 40, 144)          576       ['block_2_depthwise[0][0]']   
 hNormalization)                                                                                  
                                                                                                  
 block_2_depthwise_relu (Re  (None, 40, 40, 144)          0         ['block_2_depthwise_BN[0][0]']
 LU)                                                                                              
                                                                                                  
 block_2_project (Conv2D)    (None, 40, 40, 24)           3456      ['block_2_depthwise_relu[0][0]
                                                                    ']                            
                                                                                                  
 block_2_project_BN (BatchN  (None, 40, 40, 24)           96        ['block_2_project[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_2_add (Add)           (None, 40, 40, 24)           0         ['block_1_project_BN[0][0]',  
                                                                     'block_2_project_BN[0][0]']  
                                                                                                  
 block_3_expand (Conv2D)     (None, 40, 40, 144)          3456      ['block_2_add[0][0]']         
                                                                                                  
 block_3_expand_BN (BatchNo  (None, 40, 40, 144)          576       ['block_3_expand[0][0]']      
 rmalization)                                                                                     
                                                                                                  
 block_3_expand_relu (ReLU)  (None, 40, 40, 144)          0         ['block_3_expand_BN[0][0]']   
                                                                                                  
 block_3_pad (ZeroPadding2D  (None, 41, 41, 144)          0         ['block_3_expand_relu[0][0]'] 
 )                                                                                                
                                                                                                  
 block_3_depthwise (Depthwi  (None, 20, 20, 144)          1296      ['block_3_pad[0][0]']         
 seConv2D)                                                                                        
                                                                                                  
 block_3_depthwise_BN (Batc  (None, 20, 20, 144)          576       ['block_3_depthwise[0][0]']   
 hNormalization)                                                                                  
                                                                                                  
 block_3_depthwise_relu (Re  (None, 20, 20, 144)          0         ['block_3_depthwise_BN[0][0]']
 LU)                                                                                              
                                                                                                  
 block_3_project (Conv2D)    (None, 20, 20, 32)           4608      ['block_3_depthwise_relu[0][0]
                                                                    ']                            
                                                                                                  
 block_3_project_BN (BatchN  (None, 20, 20, 32)           128       ['block_3_project[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_4_expand (Conv2D)     (None, 20, 20, 192)          6144      ['block_3_project_BN[0][0]']  
                                                                                                  
 block_4_expand_BN (BatchNo  (None, 20, 20, 192)          768       ['block_4_expand[0][0]']      
 rmalization)                                                                                     
                                                                                                  
 block_4_expand_relu (ReLU)  (None, 20, 20, 192)          0         ['block_4_expand_BN[0][0]']   
                                                                                                  
 block_4_depthwise (Depthwi  (None, 20, 20, 192)          1728      ['block_4_expand_relu[0][0]'] 
 seConv2D)                                                                                        
                                                                                                  
 block_4_depthwise_BN (Batc  (None, 20, 20, 192)          768       ['block_4_depthwise[0][0]']   
 hNormalization)                                                                                  
                                                                                                  
 block_4_depthwise_relu (Re  (None, 20, 20, 192)          0         ['block_4_depthwise_BN[0][0]']
 LU)                                                                                              
                                                                                                  
 block_4_project (Conv2D)    (None, 20, 20, 32)           6144      ['block_4_depthwise_relu[0][0]
                                                                    ']                            
                                                                                                  
 block_4_project_BN (BatchN  (None, 20, 20, 32)           128       ['block_4_project[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_4_add (Add)           (None, 20, 20, 32)           0         ['block_3_project_BN[0][0]',  
                                                                     'block_4_project_BN[0][0]']  
                                                                                                  
 block_5_expand (Conv2D)     (None, 20, 20, 192)          6144      ['block_4_add[0][0]']         
                                                                                                  
 block_5_expand_BN (BatchNo  (None, 20, 20, 192)          768       ['block_5_expand[0][0]']      
 rmalization)                                                                                     
                                                                                                  
 block_5_expand_relu (ReLU)  (None, 20, 20, 192)          0         ['block_5_expand_BN[0][0]']   
                                                                                                  
 block_5_depthwise (Depthwi  (None, 20, 20, 192)          1728      ['block_5_expand_relu[0][0]'] 
 seConv2D)                                                                                        
                                                                                                  
 block_5_depthwise_BN (Batc  (None, 20, 20, 192)          768       ['block_5_depthwise[0][0]']   
 hNormalization)                                                                                  
                                                                                                  
 block_5_depthwise_relu (Re  (None, 20, 20, 192)          0         ['block_5_depthwise_BN[0][0]']
 LU)                                                                                              
                                                                                                  
 block_5_project (Conv2D)    (None, 20, 20, 32)           6144      ['block_5_depthwise_relu[0][0]
                                                                    ']                            
                                                                                                  
 block_5_project_BN (BatchN  (None, 20, 20, 32)           128       ['block_5_project[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_5_add (Add)           (None, 20, 20, 32)           0         ['block_4_add[0][0]',         
                                                                     'block_5_project_BN[0][0]']  
                                                                                                  
 block_6_expand (Conv2D)     (None, 20, 20, 192)          6144      ['block_5_add[0][0]']         
                                                                                                  
 block_6_expand_BN (BatchNo  (None, 20, 20, 192)          768       ['block_6_expand[0][0]']      
 rmalization)                                                                                     
                                                                                                  
 block_6_expand_relu (ReLU)  (None, 20, 20, 192)          0         ['block_6_expand_BN[0][0]']   
                                                                                                  
 block_6_pad (ZeroPadding2D  (None, 21, 21, 192)          0         ['block_6_expand_relu[0][0]'] 
 )                                                                                                
                                                                                                  
 block_6_depthwise (Depthwi  (None, 10, 10, 192)          1728      ['block_6_pad[0][0]']         
 seConv2D)                                                                                        
                                                                                                  
 block_6_depthwise_BN (Batc  (None, 10, 10, 192)          768       ['block_6_depthwise[0][0]']   
 hNormalization)                                                                                  
                                                                                                  
 block_6_depthwise_relu (Re  (None, 10, 10, 192)          0         ['block_6_depthwise_BN[0][0]']
 LU)                                                                                              
                                                                                                  
 block_6_project (Conv2D)    (None, 10, 10, 64)           12288     ['block_6_depthwise_relu[0][0]
                                                                    ']                            
                                                                                                  
 block_6_project_BN (BatchN  (None, 10, 10, 64)           256       ['block_6_project[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_7_expand (Conv2D)     (None, 10, 10, 384)          24576     ['block_6_project_BN[0][0]']  
                                                                                                  
 block_7_expand_BN (BatchNo  (None, 10, 10, 384)          1536      ['block_7_expand[0][0]']      
 rmalization)                                                                                     
                                                                                                  
 block_7_expand_relu (ReLU)  (None, 10, 10, 384)          0         ['block_7_expand_BN[0][0]']   
                                                                                                  
 block_7_depthwise (Depthwi  (None, 10, 10, 384)          3456      ['block_7_expand_relu[0][0]'] 
 seConv2D)                                                                                        
                                                                                                  
 block_7_depthwise_BN (Batc  (None, 10, 10, 384)          1536      ['block_7_depthwise[0][0]']   
 hNormalization)                                                                                  
                                                                                                  
 block_7_depthwise_relu (Re  (None, 10, 10, 384)          0         ['block_7_depthwise_BN[0][0]']
 LU)                                                                                              
                                                                                                  
 block_7_project (Conv2D)    (None, 10, 10, 64)           24576     ['block_7_depthwise_relu[0][0]
                                                                    ']                            
                                                                                                  
 block_7_project_BN (BatchN  (None, 10, 10, 64)           256       ['block_7_project[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_7_add (Add)           (None, 10, 10, 64)           0         ['block_6_project_BN[0][0]',  
                                                                     'block_7_project_BN[0][0]']  
                                                                                                  
 block_8_expand (Conv2D)     (None, 10, 10, 384)          24576     ['block_7_add[0][0]']         
                                                                                                  
 block_8_expand_BN (BatchNo  (None, 10, 10, 384)          1536      ['block_8_expand[0][0]']      
 rmalization)                                                                                     
                                                                                                  
 block_8_expand_relu (ReLU)  (None, 10, 10, 384)          0         ['block_8_expand_BN[0][0]']   
                                                                                                  
 block_8_depthwise (Depthwi  (None, 10, 10, 384)          3456      ['block_8_expand_relu[0][0]'] 
 seConv2D)                                                                                        
                                                                                                  
 block_8_depthwise_BN (Batc  (None, 10, 10, 384)          1536      ['block_8_depthwise[0][0]']   
 hNormalization)                                                                                  
                                                                                                  
 block_8_depthwise_relu (Re  (None, 10, 10, 384)          0         ['block_8_depthwise_BN[0][0]']
 LU)                                                                                              
                                                                                                  
 block_8_project (Conv2D)    (None, 10, 10, 64)           24576     ['block_8_depthwise_relu[0][0]
                                                                    ']                            
                                                                                                  
 block_8_project_BN (BatchN  (None, 10, 10, 64)           256       ['block_8_project[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_8_add (Add)           (None, 10, 10, 64)           0         ['block_7_add[0][0]',         
                                                                     'block_8_project_BN[0][0]']  
                                                                                                  
 block_9_expand (Conv2D)     (None, 10, 10, 384)          24576     ['block_8_add[0][0]']         
                                                                                                  
 block_9_expand_BN (BatchNo  (None, 10, 10, 384)          1536      ['block_9_expand[0][0]']      
 rmalization)                                                                                     
                                                                                                  
 block_9_expand_relu (ReLU)  (None, 10, 10, 384)          0         ['block_9_expand_BN[0][0]']   
                                                                                                  
 block_9_depthwise (Depthwi  (None, 10, 10, 384)          3456      ['block_9_expand_relu[0][0]'] 
 seConv2D)                                                                                        
                                                                                                  
 block_9_depthwise_BN (Batc  (None, 10, 10, 384)          1536      ['block_9_depthwise[0][0]']   
 hNormalization)                                                                                  
                                                                                                  
 block_9_depthwise_relu (Re  (None, 10, 10, 384)          0         ['block_9_depthwise_BN[0][0]']
 LU)                                                                                              
                                                                                                  
 block_9_project (Conv2D)    (None, 10, 10, 64)           24576     ['block_9_depthwise_relu[0][0]
                                                                    ']                            
                                                                                                  
 block_9_project_BN (BatchN  (None, 10, 10, 64)           256       ['block_9_project[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_9_add (Add)           (None, 10, 10, 64)           0         ['block_8_add[0][0]',         
                                                                     'block_9_project_BN[0][0]']  
                                                                                                  
 block_10_expand (Conv2D)    (None, 10, 10, 384)          24576     ['block_9_add[0][0]']         
                                                                                                  
 block_10_expand_BN (BatchN  (None, 10, 10, 384)          1536      ['block_10_expand[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_10_expand_relu (ReLU  (None, 10, 10, 384)          0         ['block_10_expand_BN[0][0]']  
 )                                                                                                
                                                                                                  
 block_10_depthwise (Depthw  (None, 10, 10, 384)          3456      ['block_10_expand_relu[0][0]']
 iseConv2D)                                                                                       
                                                                                                  
 block_10_depthwise_BN (Bat  (None, 10, 10, 384)          1536      ['block_10_depthwise[0][0]']  
 chNormalization)                                                                                 
                                                                                                  
 block_10_depthwise_relu (R  (None, 10, 10, 384)          0         ['block_10_depthwise_BN[0][0]'
 eLU)                                                               ]                             
                                                                                                  
 block_10_project (Conv2D)   (None, 10, 10, 96)           36864     ['block_10_depthwise_relu[0][0
                                                                    ]']                           
                                                                                                  
 block_10_project_BN (Batch  (None, 10, 10, 96)           384       ['block_10_project[0][0]']    
 Normalization)                                                                                   
                                                                                                  
 block_11_expand (Conv2D)    (None, 10, 10, 576)          55296     ['block_10_project_BN[0][0]'] 
                                                                                                  
 block_11_expand_BN (BatchN  (None, 10, 10, 576)          2304      ['block_11_expand[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_11_expand_relu (ReLU  (None, 10, 10, 576)          0         ['block_11_expand_BN[0][0]']  
 )                                                                                                
                                                                                                  
 block_11_depthwise (Depthw  (None, 10, 10, 576)          5184      ['block_11_expand_relu[0][0]']
 iseConv2D)                                                                                       
                                                                                                  
 block_11_depthwise_BN (Bat  (None, 10, 10, 576)          2304      ['block_11_depthwise[0][0]']  
 chNormalization)                                                                                 
                                                                                                  
 block_11_depthwise_relu (R  (None, 10, 10, 576)          0         ['block_11_depthwise_BN[0][0]'
 eLU)                                                               ]                             
                                                                                                  
 block_11_project (Conv2D)   (None, 10, 10, 96)           55296     ['block_11_depthwise_relu[0][0
                                                                    ]']                           
                                                                                                  
 block_11_project_BN (Batch  (None, 10, 10, 96)           384       ['block_11_project[0][0]']    
 Normalization)                                                                                   
                                                                                                  
 block_11_add (Add)          (None, 10, 10, 96)           0         ['block_10_project_BN[0][0]', 
                                                                     'block_11_project_BN[0][0]'] 
                                                                                                  
 block_12_expand (Conv2D)    (None, 10, 10, 576)          55296     ['block_11_add[0][0]']        
                                                                                                  
 block_12_expand_BN (BatchN  (None, 10, 10, 576)          2304      ['block_12_expand[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_12_expand_relu (ReLU  (None, 10, 10, 576)          0         ['block_12_expand_BN[0][0]']  
 )                                                                                                
                                                                                                  
 block_12_depthwise (Depthw  (None, 10, 10, 576)          5184      ['block_12_expand_relu[0][0]']
 iseConv2D)                                                                                       
                                                                                                  
 block_12_depthwise_BN (Bat  (None, 10, 10, 576)          2304      ['block_12_depthwise[0][0]']  
 chNormalization)                                                                                 
                                                                                                  
 block_12_depthwise_relu (R  (None, 10, 10, 576)          0         ['block_12_depthwise_BN[0][0]'
 eLU)                                                               ]                             
                                                                                                  
 block_12_project (Conv2D)   (None, 10, 10, 96)           55296     ['block_12_depthwise_relu[0][0
                                                                    ]']                           
                                                                                                  
 block_12_project_BN (Batch  (None, 10, 10, 96)           384       ['block_12_project[0][0]']    
 Normalization)                                                                                   
                                                                                                  
 block_12_add (Add)          (None, 10, 10, 96)           0         ['block_11_add[0][0]',        
                                                                     'block_12_project_BN[0][0]'] 
                                                                                                  
 block_13_expand (Conv2D)    (None, 10, 10, 576)          55296     ['block_12_add[0][0]']        
                                                                                                  
 block_13_expand_BN (BatchN  (None, 10, 10, 576)          2304      ['block_13_expand[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_13_expand_relu (ReLU  (None, 10, 10, 576)          0         ['block_13_expand_BN[0][0]']  
 )                                                                                                
                                                                                                  
 block_13_pad (ZeroPadding2  (None, 11, 11, 576)          0         ['block_13_expand_relu[0][0]']
 D)                                                                                               
                                                                                                  
 block_13_depthwise (Depthw  (None, 5, 5, 576)            5184      ['block_13_pad[0][0]']        
 iseConv2D)                                                                                       
                                                                                                  
 block_13_depthwise_BN (Bat  (None, 5, 5, 576)            2304      ['block_13_depthwise[0][0]']  
 chNormalization)                                                                                 
                                                                                                  
 block_13_depthwise_relu (R  (None, 5, 5, 576)            0         ['block_13_depthwise_BN[0][0]'
 eLU)                                                               ]                             
                                                                                                  
 block_13_project (Conv2D)   (None, 5, 5, 160)            92160     ['block_13_depthwise_relu[0][0
                                                                    ]']                           
                                                                                                  
 block_13_project_BN (Batch  (None, 5, 5, 160)            640       ['block_13_project[0][0]']    
 Normalization)                                                                                   
                                                                                                  
 block_14_expand (Conv2D)    (None, 5, 5, 960)            153600    ['block_13_project_BN[0][0]'] 
                                                                                                  
 block_14_expand_BN (BatchN  (None, 5, 5, 960)            3840      ['block_14_expand[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_14_expand_relu (ReLU  (None, 5, 5, 960)            0         ['block_14_expand_BN[0][0]']  
 )                                                                                                
                                                                                                  
 block_14_depthwise (Depthw  (None, 5, 5, 960)            8640      ['block_14_expand_relu[0][0]']
 iseConv2D)                                                                                       
                                                                                                  
 block_14_depthwise_BN (Bat  (None, 5, 5, 960)            3840      ['block_14_depthwise[0][0]']  
 chNormalization)                                                                                 
                                                                                                  
 block_14_depthwise_relu (R  (None, 5, 5, 960)            0         ['block_14_depthwise_BN[0][0]'
 eLU)                                                               ]                             
                                                                                                  
 block_14_project (Conv2D)   (None, 5, 5, 160)            153600    ['block_14_depthwise_relu[0][0
                                                                    ]']                           
                                                                                                  
 block_14_project_BN (Batch  (None, 5, 5, 160)            640       ['block_14_project[0][0]']    
 Normalization)                                                                                   
                                                                                                  
 block_14_add (Add)          (None, 5, 5, 160)            0         ['block_13_project_BN[0][0]', 
                                                                     'block_14_project_BN[0][0]'] 
                                                                                                  
 block_15_expand (Conv2D)    (None, 5, 5, 960)            153600    ['block_14_add[0][0]']        
                                                                                                  
 block_15_expand_BN (BatchN  (None, 5, 5, 960)            3840      ['block_15_expand[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_15_expand_relu (ReLU  (None, 5, 5, 960)            0         ['block_15_expand_BN[0][0]']  
 )                                                                                                
                                                                                                  
 block_15_depthwise (Depthw  (None, 5, 5, 960)            8640      ['block_15_expand_relu[0][0]']
 iseConv2D)                                                                                       
                                                                                                  
 block_15_depthwise_BN (Bat  (None, 5, 5, 960)            3840      ['block_15_depthwise[0][0]']  
 chNormalization)                                                                                 
                                                                                                  
 block_15_depthwise_relu (R  (None, 5, 5, 960)            0         ['block_15_depthwise_BN[0][0]'
 eLU)                                                               ]                             
                                                                                                  
 block_15_project (Conv2D)   (None, 5, 5, 160)            153600    ['block_15_depthwise_relu[0][0
                                                                    ]']                           
                                                                                                  
 block_15_project_BN (Batch  (None, 5, 5, 160)            640       ['block_15_project[0][0]']    
 Normalization)                                                                                   
                                                                                                  
 block_15_add (Add)          (None, 5, 5, 160)            0         ['block_14_add[0][0]',        
                                                                     'block_15_project_BN[0][0]'] 
                                                                                                  
 block_16_expand (Conv2D)    (None, 5, 5, 960)            153600    ['block_15_add[0][0]']        
                                                                                                  
 block_16_expand_BN (BatchN  (None, 5, 5, 960)            3840      ['block_16_expand[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_16_expand_relu (ReLU  (None, 5, 5, 960)            0         ['block_16_expand_BN[0][0]']  
 )                                                                                                
                                                                                                  
 block_16_depthwise (Depthw  (None, 5, 5, 960)            8640      ['block_16_expand_relu[0][0]']
 iseConv2D)                                                                                       
                                                                                                  
 block_16_depthwise_BN (Bat  (None, 5, 5, 960)            3840      ['block_16_depthwise[0][0]']  
 chNormalization)                                                                                 
                                                                                                  
 block_16_depthwise_relu (R  (None, 5, 5, 960)            0         ['block_16_depthwise_BN[0][0]'
 eLU)                                                               ]                             
                                                                                                  
 block_16_project (Conv2D)   (None, 5, 5, 320)            307200    ['block_16_depthwise_relu[0][0
                                                                    ]']                           
                                                                                                  
 block_16_project_BN (Batch  (None, 5, 5, 320)            1280      ['block_16_project[0][0]']    
 Normalization)                                                                                   
                                                                                                  
 Conv_1 (Conv2D)             (None, 5, 5, 1280)           409600    ['block_16_project_BN[0][0]'] 
                                                                                                  
 Conv_1_bn (BatchNormalizat  (None, 5, 5, 1280)           5120      ['Conv_1[0][0]']              
 ion)                                                                                             
                                                                                                  
 out_relu (ReLU)             (None, 5, 5, 1280)           0         ['Conv_1_bn[0][0]']           #user can take these features of 32x5x5x1280 Tensor shape received
                                                                                                  
==================================================================================================
Total params: 2257984 (8.61 MB)
Trainable params: 2223872 (8.48 MB)
Non-trainable params: 34112 (133.25 KB)
__________________________________________________________________________________________________

At this point this base_model will simply output a shape (32, 5, 5, 1280) tensor that is a feature extraction from our original (1, 160, 160, 3) image. The 32 means that we have 32 layers of differnt filters/features.


for image, _ in train_batches.take(1):
   pass

feature_batch = base_model(image)
print(feature_batch.shape)
Run
(32, 5, 5, 1280)

Freezing the Base
The term freezing refers to disabling the training property of a layer. It simply means we won’t make any changes to the weights of any layers that are frozen during training. This is important as we don't want to change the convolutional base that already has learned weights.


base_model.trainable = False #Turn training attribute of layer off


base_model.summary()
Model: "mobilenetv2_1.00_160"
__________________________________________________________________________________________________
 Layer (type)                Output Shape                 Param #   Connected to                  
==================================================================================================
 input_1 (InputLayer)        [(None, 160, 160, 3)]        0         []                            
                                                                                                  
 Conv1 (Conv2D)              (None, 80, 80, 32)           864       ['input_1[0][0]']             
                                                                                                  
 bn_Conv1 (BatchNormalizati  (None, 80, 80, 32)           128       ['Conv1[0][0]']               
 on)                                                                                              
                                                                                                  
 Conv1_relu (ReLU)           (None, 80, 80, 32)           0         ['bn_Conv1[0][0]']            
                                                                                                  
 expanded_conv_depthwise (D  (None, 80, 80, 32)           288       ['Conv1_relu[0][0]']          
 epthwiseConv2D)                                                                                  
                                                                                                  
 expanded_conv_depthwise_BN  (None, 80, 80, 32)           128       ['expanded_conv_depthwise[0][0
  (BatchNormalization)                                              ]']                           
                                                                                                  
 expanded_conv_depthwise_re  (None, 80, 80, 32)           0         ['expanded_conv_depthwise_BN[0
 lu (ReLU)                                                          ][0]']                        
                                                                                                  
 expanded_conv_project (Con  (None, 80, 80, 16)           512       ['expanded_conv_depthwise_relu
 v2D)                                                               [0][0]']                      
                                                                                                  
 expanded_conv_project_BN (  (None, 80, 80, 16)           64        ['expanded_conv_project[0][0]'
 BatchNormalization)                                                ]                             
                                                                                                  
 block_1_expand (Conv2D)     (None, 80, 80, 96)           1536      ['expanded_conv_project_BN[0][
                                                                    0]']                          
                                                                                                  
 block_1_expand_BN (BatchNo  (None, 80, 80, 96)           384       ['block_1_expand[0][0]']      
 rmalization)                                                                                     
                                                                                                  
 block_1_expand_relu (ReLU)  (None, 80, 80, 96)           0         ['block_1_expand_BN[0][0]']   
                                                                                                  
 block_1_pad (ZeroPadding2D  (None, 81, 81, 96)           0         ['block_1_expand_relu[0][0]'] 
 )                                                                                                
                                                                                                  
 block_1_depthwise (Depthwi  (None, 40, 40, 96)           864       ['block_1_pad[0][0]']         
 seConv2D)                                                                                        
                                                                                                  
 block_1_depthwise_BN (Batc  (None, 40, 40, 96)           384       ['block_1_depthwise[0][0]']   
 hNormalization)                                                                                  
                                                                                                  
 block_1_depthwise_relu (Re  (None, 40, 40, 96)           0         ['block_1_depthwise_BN[0][0]']
 LU)                                                                                              
                                                                                                  
 block_1_project (Conv2D)    (None, 40, 40, 24)           2304      ['block_1_depthwise_relu[0][0]
                                                                    ']                            
                                                                                                  
 block_1_project_BN (BatchN  (None, 40, 40, 24)           96        ['block_1_project[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_2_expand (Conv2D)     (None, 40, 40, 144)          3456      ['block_1_project_BN[0][0]']  
                                                                                                  
 block_2_expand_BN (BatchNo  (None, 40, 40, 144)          576       ['block_2_expand[0][0]']      
 rmalization)                                                                                     
                                                                                                  
 block_2_expand_relu (ReLU)  (None, 40, 40, 144)          0         ['block_2_expand_BN[0][0]']   
                                                                                                  
 block_2_depthwise (Depthwi  (None, 40, 40, 144)          1296      ['block_2_expand_relu[0][0]'] 
 seConv2D)                                                                                        
                                                                                                  
 block_2_depthwise_BN (Batc  (None, 40, 40, 144)          576       ['block_2_depthwise[0][0]']   
 hNormalization)                                                                                  
                                                                                                  
 block_2_depthwise_relu (Re  (None, 40, 40, 144)          0         ['block_2_depthwise_BN[0][0]']
 LU)                                                                                              
                                                                                                  
 block_2_project (Conv2D)    (None, 40, 40, 24)           3456      ['block_2_depthwise_relu[0][0]
                                                                    ']                            
                                                                                                  
 block_2_project_BN (BatchN  (None, 40, 40, 24)           96        ['block_2_project[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_2_add (Add)           (None, 40, 40, 24)           0         ['block_1_project_BN[0][0]',  
                                                                     'block_2_project_BN[0][0]']  
                                                                                                  
 block_3_expand (Conv2D)     (None, 40, 40, 144)          3456      ['block_2_add[0][0]']         
                                                                                                  
 block_3_expand_BN (BatchNo  (None, 40, 40, 144)          576       ['block_3_expand[0][0]']      
 rmalization)                                                                                     
                                                                                                  
 block_3_expand_relu (ReLU)  (None, 40, 40, 144)          0         ['block_3_expand_BN[0][0]']   
                                                                                                  
 block_3_pad (ZeroPadding2D  (None, 41, 41, 144)          0         ['block_3_expand_relu[0][0]'] 
 )                                                                                                
                                                                                                  
 block_3_depthwise (Depthwi  (None, 20, 20, 144)          1296      ['block_3_pad[0][0]']         
 seConv2D)                                                                                        
                                                                                                  
 block_3_depthwise_BN (Batc  (None, 20, 20, 144)          576       ['block_3_depthwise[0][0]']   
 hNormalization)                                                                                  
                                                                                                  
 block_3_depthwise_relu (Re  (None, 20, 20, 144)          0         ['block_3_depthwise_BN[0][0]']
 LU)                                                                                              
                                                                                                  
 block_3_project (Conv2D)    (None, 20, 20, 32)           4608      ['block_3_depthwise_relu[0][0]
                                                                    ']                            
                                                                                                  
 block_3_project_BN (BatchN  (None, 20, 20, 32)           128       ['block_3_project[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_4_expand (Conv2D)     (None, 20, 20, 192)          6144      ['block_3_project_BN[0][0]']  
                                                                                                  
 block_4_expand_BN (BatchNo  (None, 20, 20, 192)          768       ['block_4_expand[0][0]']      
 rmalization)                                                                                     
                                                                                                  
 block_4_expand_relu (ReLU)  (None, 20, 20, 192)          0         ['block_4_expand_BN[0][0]']   
                                                                                                  
 block_4_depthwise (Depthwi  (None, 20, 20, 192)          1728      ['block_4_expand_relu[0][0]'] 
 seConv2D)                                                                                        
                                                                                                  
 block_4_depthwise_BN (Batc  (None, 20, 20, 192)          768       ['block_4_depthwise[0][0]']   
 hNormalization)                                                                                  
                                                                                                  
 block_4_depthwise_relu (Re  (None, 20, 20, 192)          0         ['block_4_depthwise_BN[0][0]']
 LU)                                                                                              
                                                                                                  
 block_4_project (Conv2D)    (None, 20, 20, 32)           6144      ['block_4_depthwise_relu[0][0]
                                                                    ']                            
                                                                                                  
 block_4_project_BN (BatchN  (None, 20, 20, 32)           128       ['block_4_project[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_4_add (Add)           (None, 20, 20, 32)           0         ['block_3_project_BN[0][0]',  
                                                                     'block_4_project_BN[0][0]']  
                                                                                                  
 block_5_expand (Conv2D)     (None, 20, 20, 192)          6144      ['block_4_add[0][0]']         
                                                                                                  
 block_5_expand_BN (BatchNo  (None, 20, 20, 192)          768       ['block_5_expand[0][0]']      
 rmalization)                                                                                     
                                                                                                  
 block_5_expand_relu (ReLU)  (None, 20, 20, 192)          0         ['block_5_expand_BN[0][0]']   
                                                                                                  
 block_5_depthwise (Depthwi  (None, 20, 20, 192)          1728      ['block_5_expand_relu[0][0]'] 
 seConv2D)                                                                                        
                                                                                                  
 block_5_depthwise_BN (Batc  (None, 20, 20, 192)          768       ['block_5_depthwise[0][0]']   
 hNormalization)                                                                                  
                                                                                                  
 block_5_depthwise_relu (Re  (None, 20, 20, 192)          0         ['block_5_depthwise_BN[0][0]']
 LU)                                                                                              
                                                                                                  
 block_5_project (Conv2D)    (None, 20, 20, 32)           6144      ['block_5_depthwise_relu[0][0]
                                                                    ']                            
                                                                                                  
 block_5_project_BN (BatchN  (None, 20, 20, 32)           128       ['block_5_project[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_5_add (Add)           (None, 20, 20, 32)           0         ['block_4_add[0][0]',         
                                                                     'block_5_project_BN[0][0]']  
                                                                                                  
 block_6_expand (Conv2D)     (None, 20, 20, 192)          6144      ['block_5_add[0][0]']         
                                                                                                  
 block_6_expand_BN (BatchNo  (None, 20, 20, 192)          768       ['block_6_expand[0][0]']      
 rmalization)                                                                                     
                                                                                                  
 block_6_expand_relu (ReLU)  (None, 20, 20, 192)          0         ['block_6_expand_BN[0][0]']   
                                                                                                  
 block_6_pad (ZeroPadding2D  (None, 21, 21, 192)          0         ['block_6_expand_relu[0][0]'] 
 )                                                                                                
                                                                                                  
 block_6_depthwise (Depthwi  (None, 10, 10, 192)          1728      ['block_6_pad[0][0]']         
 seConv2D)                                                                                        
                                                                                                  
 block_6_depthwise_BN (Batc  (None, 10, 10, 192)          768       ['block_6_depthwise[0][0]']   
 hNormalization)                                                                                  
                                                                                                  
 block_6_depthwise_relu (Re  (None, 10, 10, 192)          0         ['block_6_depthwise_BN[0][0]']
 LU)                                                                                              
                                                                                                  
 block_6_project (Conv2D)    (None, 10, 10, 64)           12288     ['block_6_depthwise_relu[0][0]
                                                                    ']                            
                                                                                                  
 block_6_project_BN (BatchN  (None, 10, 10, 64)           256       ['block_6_project[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_7_expand (Conv2D)     (None, 10, 10, 384)          24576     ['block_6_project_BN[0][0]']  
                                                                                                  
 block_7_expand_BN (BatchNo  (None, 10, 10, 384)          1536      ['block_7_expand[0][0]']      
 rmalization)                                                                                     
                                                                                                  
 block_7_expand_relu (ReLU)  (None, 10, 10, 384)          0         ['block_7_expand_BN[0][0]']   
                                                                                                  
 block_7_depthwise (Depthwi  (None, 10, 10, 384)          3456      ['block_7_expand_relu[0][0]'] 
 seConv2D)                                                                                        
                                                                                                  
 block_7_depthwise_BN (Batc  (None, 10, 10, 384)          1536      ['block_7_depthwise[0][0]']   
 hNormalization)                                                                                  
                                                                                                  
 block_7_depthwise_relu (Re  (None, 10, 10, 384)          0         ['block_7_depthwise_BN[0][0]']
 LU)                                                                                              
                                                                                                  
 block_7_project (Conv2D)    (None, 10, 10, 64)           24576     ['block_7_depthwise_relu[0][0]
                                                                    ']                            
                                                                                                  
 block_7_project_BN (BatchN  (None, 10, 10, 64)           256       ['block_7_project[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_7_add (Add)           (None, 10, 10, 64)           0         ['block_6_project_BN[0][0]',  
                                                                     'block_7_project_BN[0][0]']  
                                                                                                  
 block_8_expand (Conv2D)     (None, 10, 10, 384)          24576     ['block_7_add[0][0]']         
                                                                                                  
 block_8_expand_BN (BatchNo  (None, 10, 10, 384)          1536      ['block_8_expand[0][0]']      
 rmalization)                                                                                     
                                                                                                  
 block_8_expand_relu (ReLU)  (None, 10, 10, 384)          0         ['block_8_expand_BN[0][0]']   
                                                                                                  
 block_8_depthwise (Depthwi  (None, 10, 10, 384)          3456      ['block_8_expand_relu[0][0]'] 
 seConv2D)                                                                                        
                                                                                                  
 block_8_depthwise_BN (Batc  (None, 10, 10, 384)          1536      ['block_8_depthwise[0][0]']   
 hNormalization)                                                                                  
                                                                                                  
 block_8_depthwise_relu (Re  (None, 10, 10, 384)          0         ['block_8_depthwise_BN[0][0]']
 LU)                                                                                              
                                                                                                  
 block_8_project (Conv2D)    (None, 10, 10, 64)           24576     ['block_8_depthwise_relu[0][0]
                                                                    ']                            
                                                                                                  
 block_8_project_BN (BatchN  (None, 10, 10, 64)           256       ['block_8_project[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_8_add (Add)           (None, 10, 10, 64)           0         ['block_7_add[0][0]',         
                                                                     'block_8_project_BN[0][0]']  
                                                                                                  
 block_9_expand (Conv2D)     (None, 10, 10, 384)          24576     ['block_8_add[0][0]']         
                                                                                                  
 block_9_expand_BN (BatchNo  (None, 10, 10, 384)          1536      ['block_9_expand[0][0]']      
 rmalization)                                                                                     
                                                                                                  
 block_9_expand_relu (ReLU)  (None, 10, 10, 384)          0         ['block_9_expand_BN[0][0]']   
                                                                                                  
 block_9_depthwise (Depthwi  (None, 10, 10, 384)          3456      ['block_9_expand_relu[0][0]'] 
 seConv2D)                                                                                        
                                                                                                  
 block_9_depthwise_BN (Batc  (None, 10, 10, 384)          1536      ['block_9_depthwise[0][0]']   
 hNormalization)                                                                                  
                                                                                                  
 block_9_depthwise_relu (Re  (None, 10, 10, 384)          0         ['block_9_depthwise_BN[0][0]']
 LU)                                                                                              
                                                                                                  
 block_9_project (Conv2D)    (None, 10, 10, 64)           24576     ['block_9_depthwise_relu[0][0]
                                                                    ']                            
                                                                                                  
 block_9_project_BN (BatchN  (None, 10, 10, 64)           256       ['block_9_project[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_9_add (Add)           (None, 10, 10, 64)           0         ['block_8_add[0][0]',         
                                                                     'block_9_project_BN[0][0]']  
                                                                                                  
 block_10_expand (Conv2D)    (None, 10, 10, 384)          24576     ['block_9_add[0][0]']         
                                                                                                  
 block_10_expand_BN (BatchN  (None, 10, 10, 384)          1536      ['block_10_expand[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_10_expand_relu (ReLU  (None, 10, 10, 384)          0         ['block_10_expand_BN[0][0]']  
 )                                                                                                
                                                                                                  
 block_10_depthwise (Depthw  (None, 10, 10, 384)          3456      ['block_10_expand_relu[0][0]']
 iseConv2D)                                                                                       
                                                                                                  
 block_10_depthwise_BN (Bat  (None, 10, 10, 384)          1536      ['block_10_depthwise[0][0]']  
 chNormalization)                                                                                 
                                                                                                  
 block_10_depthwise_relu (R  (None, 10, 10, 384)          0         ['block_10_depthwise_BN[0][0]'
 eLU)                                                               ]                             
                                                                                                  
 block_10_project (Conv2D)   (None, 10, 10, 96)           36864     ['block_10_depthwise_relu[0][0
                                                                    ]']                           
                                                                                                  
 block_10_project_BN (Batch  (None, 10, 10, 96)           384       ['block_10_project[0][0]']    
 Normalization)                                                                                   
                                                                                                  
 block_11_expand (Conv2D)    (None, 10, 10, 576)          55296     ['block_10_project_BN[0][0]'] 
                                                                                                  
 block_11_expand_BN (BatchN  (None, 10, 10, 576)          2304      ['block_11_expand[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_11_expand_relu (ReLU  (None, 10, 10, 576)          0         ['block_11_expand_BN[0][0]']  
 )                                                                                                
                                                                                                  
 block_11_depthwise (Depthw  (None, 10, 10, 576)          5184      ['block_11_expand_relu[0][0]']
 iseConv2D)                                                                                       
                                                                                                  
 block_11_depthwise_BN (Bat  (None, 10, 10, 576)          2304      ['block_11_depthwise[0][0]']  
 chNormalization)                                                                                 
                                                                                                  
 block_11_depthwise_relu (R  (None, 10, 10, 576)          0         ['block_11_depthwise_BN[0][0]'
 eLU)                                                               ]                             
                                                                                                  
 block_11_project (Conv2D)   (None, 10, 10, 96)           55296     ['block_11_depthwise_relu[0][0
                                                                    ]']                           
                                                                                                  
 block_11_project_BN (Batch  (None, 10, 10, 96)           384       ['block_11_project[0][0]']    
 Normalization)                                                                                   
                                                                                                  
 block_11_add (Add)          (None, 10, 10, 96)           0         ['block_10_project_BN[0][0]', 
                                                                     'block_11_project_BN[0][0]'] 
                                                                                                  
 block_12_expand (Conv2D)    (None, 10, 10, 576)          55296     ['block_11_add[0][0]']        
                                                                                                  
 block_12_expand_BN (BatchN  (None, 10, 10, 576)          2304      ['block_12_expand[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_12_expand_relu (ReLU  (None, 10, 10, 576)          0         ['block_12_expand_BN[0][0]']  
 )                                                                                                
                                                                                                  
 block_12_depthwise (Depthw  (None, 10, 10, 576)          5184      ['block_12_expand_relu[0][0]']
 iseConv2D)                                                                                       
                                                                                                  
 block_12_depthwise_BN (Bat  (None, 10, 10, 576)          2304      ['block_12_depthwise[0][0]']  
 chNormalization)                                                                                 
                                                                                                  
 block_12_depthwise_relu (R  (None, 10, 10, 576)          0         ['block_12_depthwise_BN[0][0]'
 eLU)                                                               ]                             
                                                                                                  
 block_12_project (Conv2D)   (None, 10, 10, 96)           55296     ['block_12_depthwise_relu[0][0
                                                                    ]']                           
                                                                                                  
 block_12_project_BN (Batch  (None, 10, 10, 96)           384       ['block_12_project[0][0]']    
 Normalization)                                                                                   
                                                                                                  
 block_12_add (Add)          (None, 10, 10, 96)           0         ['block_11_add[0][0]',        
                                                                     'block_12_project_BN[0][0]'] 
                                                                                                  
 block_13_expand (Conv2D)    (None, 10, 10, 576)          55296     ['block_12_add[0][0]']        
                                                                                                  
 block_13_expand_BN (BatchN  (None, 10, 10, 576)          2304      ['block_13_expand[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_13_expand_relu (ReLU  (None, 10, 10, 576)          0         ['block_13_expand_BN[0][0]']  
 )                                                                                                
                                                                                                  
 block_13_pad (ZeroPadding2  (None, 11, 11, 576)          0         ['block_13_expand_relu[0][0]']
 D)                                                                                               
                                                                                                  
 block_13_depthwise (Depthw  (None, 5, 5, 576)            5184      ['block_13_pad[0][0]']        
 iseConv2D)                                                                                       
                                                                                                  
 block_13_depthwise_BN (Bat  (None, 5, 5, 576)            2304      ['block_13_depthwise[0][0]']  
 chNormalization)                                                                                 
                                                                                                  
 block_13_depthwise_relu (R  (None, 5, 5, 576)            0         ['block_13_depthwise_BN[0][0]'
 eLU)                                                               ]                             
                                                                                                  
 block_13_project (Conv2D)   (None, 5, 5, 160)            92160     ['block_13_depthwise_relu[0][0
                                                                    ]']                           
                                                                                                  
 block_13_project_BN (Batch  (None, 5, 5, 160)            640       ['block_13_project[0][0]']    
 Normalization)                                                                                   
                                                                                                  
 block_14_expand (Conv2D)    (None, 5, 5, 960)            153600    ['block_13_project_BN[0][0]'] 
                                                                                                  
 block_14_expand_BN (BatchN  (None, 5, 5, 960)            3840      ['block_14_expand[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_14_expand_relu (ReLU  (None, 5, 5, 960)            0         ['block_14_expand_BN[0][0]']  
 )                                                                                                
                                                                                                  
 block_14_depthwise (Depthw  (None, 5, 5, 960)            8640      ['block_14_expand_relu[0][0]']
 iseConv2D)                                                                                       
                                                                                                  
 block_14_depthwise_BN (Bat  (None, 5, 5, 960)            3840      ['block_14_depthwise[0][0]']  
 chNormalization)                                                                                 
                                                                                                  
 block_14_depthwise_relu (R  (None, 5, 5, 960)            0         ['block_14_depthwise_BN[0][0]'
 eLU)                                                               ]                             
                                                                                                  
 block_14_project (Conv2D)   (None, 5, 5, 160)            153600    ['block_14_depthwise_relu[0][0
                                                                    ]']                           
                                                                                                  
 block_14_project_BN (Batch  (None, 5, 5, 160)            640       ['block_14_project[0][0]']    
 Normalization)                                                                                   
                                                                                                  
 block_14_add (Add)          (None, 5, 5, 160)            0         ['block_13_project_BN[0][0]', 
                                                                     'block_14_project_BN[0][0]'] 
                                                                                                  
 block_15_expand (Conv2D)    (None, 5, 5, 960)            153600    ['block_14_add[0][0]']        
                                                                                                  
 block_15_expand_BN (BatchN  (None, 5, 5, 960)            3840      ['block_15_expand[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_15_expand_relu (ReLU  (None, 5, 5, 960)            0         ['block_15_expand_BN[0][0]']  
 )                                                                                                
                                                                                                  
 block_15_depthwise (Depthw  (None, 5, 5, 960)            8640      ['block_15_expand_relu[0][0]']
 iseConv2D)                                                                                       
                                                                                                  
 block_15_depthwise_BN (Bat  (None, 5, 5, 960)            3840      ['block_15_depthwise[0][0]']  
 chNormalization)                                                                                 
                                                                                                  
 block_15_depthwise_relu (R  (None, 5, 5, 960)            0         ['block_15_depthwise_BN[0][0]'
 eLU)                                                               ]                             
                                                                                                  
 block_15_project (Conv2D)   (None, 5, 5, 160)            153600    ['block_15_depthwise_relu[0][0
                                                                    ]']                           
                                                                                                  
 block_15_project_BN (Batch  (None, 5, 5, 160)            640       ['block_15_project[0][0]']    
 Normalization)                                                                                   
                                                                                                  
 block_15_add (Add)          (None, 5, 5, 160)            0         ['block_14_add[0][0]',        
                                                                     'block_15_project_BN[0][0]'] 
                                                                                                  
 block_16_expand (Conv2D)    (None, 5, 5, 960)            153600    ['block_15_add[0][0]']        
                                                                                                  
 block_16_expand_BN (BatchN  (None, 5, 5, 960)            3840      ['block_16_expand[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_16_expand_relu (ReLU  (None, 5, 5, 960)            0         ['block_16_expand_BN[0][0]']  
 )                                                                                                
                                                                                                  
 block_16_depthwise (Depthw  (None, 5, 5, 960)            8640      ['block_16_expand_relu[0][0]']
 iseConv2D)                                                                                       
                                                                                                  
 block_16_depthwise_BN (Bat  (None, 5, 5, 960)            3840      ['block_16_depthwise[0][0]']  
 chNormalization)                                                                                 
                                                                                                  
 block_16_depthwise_relu (R  (None, 5, 5, 960)            0         ['block_16_depthwise_BN[0][0]'
 eLU)                                                               ]                             
                                                                                                  
 block_16_project (Conv2D)   (None, 5, 5, 320)            307200    ['block_16_depthwise_relu[0][0
                                                                    ]']                           
                                                                                                  
 block_16_project_BN (Batch  (None, 5, 5, 320)            1280      ['block_16_project[0][0]']    
 Normalization)                                                                                   
                                                                                                  
 Conv_1 (Conv2D)             (None, 5, 5, 1280)           409600    ['block_16_project_BN[0][0]'] 
                                                                                                  
 Conv_1_bn (BatchNormalizat  (None, 5, 5, 1280)           5120      ['Conv_1[0][0]']              
 ion)                                                                                             
                                                                                                  
 out_relu (ReLU)             (None, 5, 5, 1280)           0         ['Conv_1_bn[0][0]']           
                                                                                                  
==================================================================================================
Total params: 2257984 (8.61 MB)
Trainable params: 0 (0.00 Byte) #before 2.257M
Non-trainable params: 2257984 (8.61 MB)
__________________________________________________________________________________________________



Adding our Classifier
Now that we have our base layer setup, we can add the classifier. Instead of flattening the feature map of the base layer we will use a global average pooling layer that will average the entire 5x5 area of each 2D feature map and return to us a single 1280 element vector per filter.


global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
Run
1d flat Tensor from above code


Finally, we will add the predicition layer that will be a single dense neuron. We can do this because we only have two classes to predict for.


prediction_layer = keras.layers.Dense(1)
Run
#1 dense node because only classifying dogs and cats


Now we will combine these layers together in a model.


model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])
#create final model


model.summary()
Run
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 mobilenetv2_1.00_160 (Func  (None, 5, 5, 1280)        2257984   #base layer
 tional)                                                         
                                                                 
 global_average_pooling2d (  (None, 1280)              0         #average
 GlobalAveragePooling2D)                                         
                                                                 
 dense_2 (Dense)             (None, 1)                 1281      #1 neuron and is output
                                                                 
=================================================================
Total params: 2259265 (8.62 MB)
Trainable params: 1281 (5.00 KB) #1280 connections from global_average_pooling2d to dense_2 and 1 bias
Non-trainable params: 2257984 (8.61 MB)
_________________________________________________________________


Training the Model
Now we will train and compile the model. We will use a very small learning rate to ensure that the model does not have any major changes made to it.


base_learning_rate = 0.0001 #how much to modify weights and biases of network to make a very low change
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate), 
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), #binary because 2 classes
              metrics=['accuracy'])



# We can evaluate the model right now to see how it does before training it on our new images
initial_epochs = 3 #
validation_steps=20

loss0,accuracy0 = model.evaluate(validation_batches, steps = validation_steps)
Run
20/20 [==============================] - 16s 734ms/step - loss: 0.7668 - accuracy: 0.4953 #50% should be guess

# Now we can train it on our images
history = model.fit(train_batches,
                    epochs=initial_epochs,
                    validation_data=validation_batches)
Run
Epoch 1/3
582/582 [==============================] - 416s 707ms/step - loss: 0.0716 - accuracy: 0.9730 - val_loss: 0.0495 - val_accuracy: 0.9845
Epoch 2/3
582/582 [==============================] - 418s 716ms/step - loss: 0.0442 - accuracy: 0.9837 - val_loss: 0.0478 - val_accuracy: 0.9824
Epoch 3/3
582/582 [==============================] - 373s 638ms/step - loss: 0.0399 - accuracy: 0.9865 - val_loss: 0.0498 - val_accuracy: 0.9802
[0.9730252623558044, 0.9837184548377991, 0.9865126013755798] #pretty good accuracy after training. Base layer classifies up to 1000 different images and is applied to only cats and dogs by adding dense layer classsifier on top.

acc = history.history['accuracy']
print(acc)


model.save("dogs_vs_cats.h5")  # we can save the model and reload it at anytime in the future specific to keras and not TensorFlow
new_model = tf.keras.models.load_model('dogs_vs_cats.h5') #load model because after an hour of training user won't want to retrain
Run
/usr/local/lib/python3.10/dist-packages/keras/src/engine/training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
  saving_api.save_model(

#typing model.pred user can see how to set up predictions input and will return if cat or dog 

And that's it for this section on computer vision!

Object Detection
If you'd like to learn how you can perform object detection and recognition with tensorflow check out the guide below.

https://github.com/tensorflow/models/tree/master/research/object_detection

Sources
“Convolutional Neural Network (CNN)  :   TensorFlow Core.” TensorFlow, www.tensorflow.org/tutorials/images/cnn.
“Transfer Learning with a Pretrained ConvNet  :   TensorFlow Core.” TensorFlow, www.tensorflow.org/tutorials/images/transfer_learning.
Chollet François. Deep Learning with Python. Manning Publications Co., 2018.


