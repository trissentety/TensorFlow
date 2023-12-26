"""
Python Neural Networks

Neural networks provide classifications or predictions for user.
Neural networks recieve input information and get an output.

Example
Black box receiving data input such as images,random data points or data set
in return get some meaningful output
Blackbox is a function of input and maps output

Another Example
Sloped Line Function
y = 4(x)
input x that gives value of y

Neural network is made up of Layer

Example
1st Layer - Input layer is 3 circles that accepts raw data which is data wanted to classify or what input information is

Further Example
Image with pixels
Classifications of image are width and height, such as a classic example of 28 x 28 px
Number of input neurons needed in neural network is 28 * 28 = 784 which is an example using input layer to represent image
1 pixel is passed to each neuron

Example 1 piece of information 
1 Number may need only 1 neuron

4 pieces of information Example
4 Neurons

1 neuron for 1 piece of info unless reshaping or putting info in different form


Output Layer
Neurons representing node in layer as output pieces

Classification for images
2 classes that could be represented
2 Ways to design output layer

Use 1 output neuron that gives some value
Value wanted is between [0, 1] Inclusive

If predicting two classes
If output is closer to 0 then it is class 0 closer to 1 is class 1
So what this means is training data when giving input the output would have to be 0 or 1
Correct class would be 0 or 1 so labels for training data set would be 0 and 1, value in ouput neuron is gauranteed to be 0 or 1

Having as many output neurons as classes looking to predict for for makes most sense
Exmaple 5 Classes
3 pieces of information to make prediction
Would have 5 output neurons and each neuron has value between 0 and 1
Combination or sum of all values would be equal to 1. What this means is it is a probability distribution
So what happens is predictions are made for how strongly each input info is of each class
So 5 classes
Class 1 is 0.9 as 90%
Class 2 is 0.001
Class 3 is 0.05
Class 4 is 0.003
Class 5
All combined add up to 1 which is probability distribution of output layer

Regression task would have 1 neuron that predicts some value and define what value would be

1 output Neuron in Output layer.
Inbetween these layers are hidden layers, hidden because not observed.
Output layer > hidden layer > output layer
Every layer is connected to another layer with weights
Connections go to different layers

Densely Connected Neural Network
Mean Every node is connected from previous layer
So every node in input layer is connected to ever node in output layer
Connections are called weights

Weights 
What the neural netowrk changes and optimizes to determine mapping from input to output
You get input and output by modifying weights
Connection lines from input to hidden is a numeric value
Typically numeric values are between 0 and 1 but can be larger or negative and it depends on type of network and how its designed

Connection line numbers
Line 1 is 0.1
Line 2 is 0.7 etc
These are known as the trainable parameters that neural network will change as it trains to get best possible result.


Hidden layer is also connected to Output Layer and is also a densely connected layer because every neauron is connected to every neuron from next layer
To determine how many connections there is
3 neurons in input layer, 2 neurons in hidden layer so 3 * 2 = 6 connections


Biases
Input Bias
Biases are loaded different than previous nodes on input layer
Only 1 bias exists and is in previous layer to layer it affects
Bias connects to each neuron in next layer and is Densely Connected but different
Bias takes no input information like input layer nodes and is another trainable parameter for the netowrk
Bias is a constant numeric value that is connected to hidden layer to do few different things with.
Weights for bias always have a value of 1 typically

Output Bias
Because there's 1 bias on input layer there is also a bias on hidden layer that connected to output Layer
Biases do not connect to eachother and reason is for constant value and bias is only something added to network as another trainable parameter that could be used.


How information is passed through network and why weights and biases are used and what they do
Example
(x, y, z)
Either part of [Red, Blue] class
What is wanted is output neuron to give red or blue
Because there's 1 class, output neuron is between range of 0 and 1
Output range
If it is closer to 0 it is red if it is closer to 1 then it is blue 
Input neurons are x for 1, y for second input neuron and z for 3rd
Data points picked for x, y, z are 2, 2, 2
How it is passed through
Determine how to find value of hidden layer node
node 1 and node 2 is equal to weighted sum of all previous nodes connected
Weighted Sum equation
i = 0 to n of w * xi + b
 n of w * xi + b means take weight sum of all neurons connected to next neuron
In this case neuron x, y, z connected to hidden neuron 1
so weighted sum calculated is really equal to weight at neuron x
wx times value at neuron x = 2
wx times value at neuron x = 2 + weight at neuron y * value at neuron x which is equal to 2 wx(2)
wx times value at neuron x = 2 + weight at neuron y * value at neuron x which is equal to 2 + weight at neuron  wx(2) + wy(2) 
wx times value at neuron x = 2 + weight at neuron y * value at neuron x which is equal to 2 + weight at neuron  wx(2) + wy(2) + wz(2)

Weights have some numeric value and when neural network is started weights are completely random values that can be used
As newtork is better, weights are updated and changed to make more sense in network
WX, WY, WZ are some numeric values

Returns
wx(2) + wy(2) + wz(2)
= V  
This bias is added and bias is connected with weight of 1 so if sum of weighted bias is taken all that happens is adding 
So if user takes sum of weighted bias
Weighted sum of bias is adding the bias value so if bias is 100 add 100 so + b which is 100 maybe
bias could be considered part of summation equation because it's another connection to neural.

Squiggly S symbol stands for sum
i stands for index
n stands for what index will go up to
n means how many neurons in previous layer

So wixi or weight 0 x 0 + weight 1 x 1 + weight 2 x 2 which is like a for loop wehere they're added together and add b

So layers are gone through and values are calculated
hidden node 1 is 0.3
hidden node 2 is 7

Output Neuron
Weighted sum of hidden node 1 * weight
Weighted sum of hidden node 2 * weight + bias
Output node is given some value and could look at value and determine what the output of neural network is.
 



Part B
Missing feature: Activation Function

Remember
Value at output layer is between 0 and 1
This can't be gauranteed with random weights and random biases passing through

When passing information through could have 700 as value
How to determine with a value as high as 700 if it is red or blue
This is done by using an Activation Function







