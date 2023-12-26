"""
https://colab.research.google.com/drive/1m2cg3D1x3j5vrFc-Cu0gMvc48gWyCOuG#scrollTo=F92rhvd6PcRI

Introduction to Neural Networks
In this notebook you will learn how to create and use a neural network to classify articles of clothing. To achieve this, we will use a sub module of TensorFlow called keras.

This guide is based on the following TensorFlow documentation.

https://www.tensorflow.org/tutorials/keras/classification

Keras
Before we dive in and start discussing neural networks, I'd like to give a breif introduction to keras.

From the keras official documentation (https://keras.io/) keras is described as follows.

"Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. It was developed with a focus on enabling fast experimentation.

Use Keras if you need a deep learning library that:

Allows for easy and fast prototyping (through user friendliness, modularity, and extensibility).
Supports both convolutional networks and recurrent networks, as well as combinations of the two.
Runs seamlessly on CPU and GPU."
Keras is a very powerful module that allows us to avoid having to build neural networks from scratch. It also hides a lot of mathematical complexity (that otherwise we would have to implement) inside of helpful packages, modules and methods.

In this guide we will use keras to quickly develop neural networks.

What is a Neural Network
So, what are these magical things that have been beating chess grandmasters, driving cars, detecting cancer cells and winning video games?

A deep neural network is a layered representation of data. The term "deep" refers to the presence of multiple layers. Recall that in our core learning algorithms (like linear regression) data was not transformed or modified within the model, it simply existed in one layer. We passed some features to our model, some math was done, an answer was returned. The data was not changed or transformed throughout this process. A neural network processes our data differently. It attempts to represent our data in different ways and in different dimensions by applying specific operations to transform our data at each layer. Another way to express this is that at each layer our data is transformed in order to learn more about it. By performing these transformations, the model can better understand our data and therefore provide a better prediction.

How it Works
Before going into too much detail I will provide a very surface level explination of how neural networks work on a mathematical level. All the terms and concepts I discuss will be defined and explained in more detail below.

On a lower level neural networks are simply a combination of elementry math operations and some more advanced linear algebra. Each neural network consists of a sequence of layers in which data passes through. These layers are made up on neurons and the neurons of one layer are connected to the next (see below). These connections are defined by what we call a weight (some numeric value). Each layer also has something called a bias, this is simply an extra neuron that has no connections and holds a single numeric value. Data starts at the input layer and is trasnformed as it passes through subsequent layers. The data at each subsequent neuron is defined as the following.

Y=(∑ni=0wixi)+b

w stands for the weight of each connection to the neuron

x stands for the value of the connected neuron from the previous value

b stands for the bias at each layer, this is a constant

n is the number of connections

Y is the output of the current neuron

∑ stands for sum

The equation you just read is called a weighed sum. We will take this weighted sum at each and every neuron as we pass information through the network. Then we will add what's called a bias to this sum. The bias allows us to shift the network up or down by a constant value. It is like the y-intercept of a line.

But that equation is the not complete one! We forgot a crucial part, the activation function. This is a function that we apply to the equation seen above to add complexity and dimensionality to our network. Our new equation with the addition of an activation function F(x) is seen below.

Y=F((∑ni=0wixi)+b)

Our network will start with predefined activation functions (they may be different at each layer) but random weights and biases. As we train the network by feeding it data it will learn the correct weights and biases and adjust the network accordingly using a technqiue called backpropagation (explained below). Once the correct weights and biases have been learned our network will hopefully be able to give us meaningful predictions. We get these predictions by observing the values at our final layer, the output layer.

Breaking Down The Neural Network!
Before we dive into any code lets break down how a neural network works and what it does.

alt text Figure 1

Data
The type of data a neural network processes varies drastically based on the problem being solved. When we build a neural network, we define what shape and kind of data it can accept. It may sometimes be neccessary to modify our dataset so that it can be passed to our neural network.

Some common types of data a neural network uses are listed below.

Vector Data (2D)
Timeseries or Sequence (3D)
Image Data (4D)
Video Data (5D)
There are of course many different types or data, but these are the main categories.

Layers
As we mentioned earlier each neural network consists of multiple layers. At each layer a different transformation of data occurs. Our initial input data is fed through the layers and eventually arrives at the output layer where we will obtain the result.

Input Layer
The input layer is the layer that our initial data is passed to. It is the first layer in our neural network.

Output Layer
The output layer is the layer that we will retrive our results from. Once the data has passed through all other layers it will arrive here.

Hidden Layer(s)
All the other layers in our neural network are called "hidden layers". This is because they are hidden to us, we cannot observe them. Most neural networks consist of at least one hidden layer but can have an unlimited amount. Typically, the more complex the model the more hidden layers.

Neurons
Each layer is made up of what are called neurons. Neurons have a few different properties that we will discuss later. The important aspect to understand now is that each neuron is responsible for generating/holding/passing ONE numeric value.

This means that in the case of our input layer it will have as many neurons as we have input information. For example, say we want to pass an image that is 28x28 pixels, thats 784 pixels. We would need 784 neurons in our input layer to capture each of these pixels.

This also means that our output layer will have as many neurons as we have output information. The output is a little more complicated to understand so I'll refrain from an example right now but hopefully you're getting the idea.

But what about our hidden layers? Well these have as many neurons as we decide. We'll discuss how we can pick these values later but understand a hidden layer can have any number of neurons.

Connected Layers
So how are all these layers connected? Well the neurons in one layer will be connected to neurons in the subsequent layer. However, the neurons can be connected in a variety of different ways.

Take for example Figure 1 (look above). Each neuron in one layer is connected to every neuron in the next layer. This is called a dense layer. There are many other ways of connecting layers but well discuss those as we see them.

Weights
Weights are associated with each connection in our neural network. Every pair of connected nodes will have one weight that denotes the strength of the connection between them. These are vital to the inner workings of a neural network and will be tweaked as the neural network is trained. The model will try to determine what these weights should be to achieve the best result. Weights start out at a constant or random value and will change as the network sees training data.

Biases
Biases are another important part of neural networks and will also be tweaked as the model is trained. A bias is simply a constant value associated with each layer. It can be thought of as an extra neuron that has no connections. The purpose of a bias is to shift an entire activation function by a constant value. This allows a lot more flexibllity when it comes to choosing an activation and training the network. There is one bias for each layer.

Activation Function
Activation functions are simply a function that is applied to the weighed sum of a neuron. They can be anything we want but are typically higher order/degree functions that aim to add a higher dimension to our data. We would want to do this to introduce more comolexity to our model. By transforming our data to a higher dimension, we can typically make better, more complex predictions.

Relu (Rectified Linear Unit)
https://yashuseth.files.wordpress.com/2018/02/relu-function.png?w=309&h=274
Takes values that are less than 0 and makes them 0
x values that are negative makes their y 0
values that are positive is equal to what their positive value is so x10 is 10
Allows user to eliminate negative numbers

Tanh (Hyperbolic Tangent)
http://mathworld.wolfram.com/images/interactive/TanhReal.gif
Squishes values between -1 and 1
Takes values and more positive values are- more closer to one, more negative values are- more closer to negative those values are


Sigmoid
https://miro.medium.com/max/970/1*Xu7B5y9gp0iL5ooBj7LtWw.png
Squish values between 0 and 1
Also known as Squishifier Function
What it does is take any string of negative numbers and put them closer to 0 and very positive numbers closer to 1
Any values inbetween gives some number between that
based on equation 1 over 1 + e to the -z


How to use them
What happens is at each neuron there's an activation function that's applied to output of that neauron
So weighted sum + bias n1=(∑0wixi)+b and apply activation function to it before sending value to next neuron.
In this case n1 is not equal to above but instead is equal to N1 = F(∑i=0 wixi + b) is what n1's value is equal to when it comes to output neuron
Hidden nodes n1 and n2 have same activation function as n1
And what can be defined is what activation function to apply at each neuron.
At the output neuron, activation function is very important because user needs to determine what the value should look like.
Such as wanting value between -1 and 1 or 0 and 1 or some large number or between 0 and postitive infinite

So what happens is user picks some activation function for output neuron
We want our values between 0 and 1
Function this time is sigmoid function. Sigmoid squishes values between 0 and 1
So here (N1 * W0 + N2 * W1 + b) = [0, 1]
Value can be looked at to determine what output of this network is.


Why to use activation function on intermediate layer like Hidden Layer
Point of activation function is to introduce complexity into the Neural Network.
Essentially there's basic weights and biases that are training and changing to make network better

Example What activation function can do
Take a bunch of points on same plane (rectangular display)
If applying activation function to this where higher dimensionality is introduced like a sigmoid, points can hopefully be spread out and move them up or down off plane in hopes of extracting different features.
This will allow for more complex predictions and pick up on different patterns that could previously not be determined.

Example looking at something in 2 dimensions
If that can be moved into 3 dimensions, more detail is seen right away.
So there's a sqaure
If asked what info can be told about square, answer could be width, height and color.
Could say it has 1 face and 4 vertexes and the area.
What happens when it is extended into a cube?
By observing, A lot more info can be told like depth, Width, height, number of faces, color of each face, how many vertexes, if cube is square or rectangular formed etc.

So those examples are an oversimplification of what activation functions do


How Neural Networks Train
Weights and biases are what network comes up with and determine to make network better.


Loss Function
As network starts, way it is training other networks or machine learning models is to give information and expected output and then see what expected output or what output was from network and compare it to expected output and modify it.

Example
So start with 2,2,2 
Class is Red as 0
user wants network to give 0 for point 2,2,2
Network starts with random weights and biases
When getting to output odds are not going to get result 0
After applying sigmoid function might get result 0.7 which is far away from red, how far away though?
This is where loss function comes into play

What Loss Function does is calculate how far away output was from expected output.
Example
If expected output is 0 and output is 0.7 then loss function will give value representing out bad or good this network was.
If it is really bad it gives a very high loss and tells user that user needs to change weights and biases more and move network in a different direction more drastically.
This is starting to get into gradient descent but loss function should be understood first.
If it is really good then only a little changing and moving of this and that.
Point of loss function is to calculate some value. Higher value, worse network was.


Examples of a loss function also classified as cost or lost functions, sometimes used interchangebly because cost and lost mean same thing.
User wants network to cost least and user wants netowrk to have least amount of loss

Mean Squared Error
Mean Absolute Error
Hinge Loss


How to update Weights and Biases

Gradient Descent

Gradient Descent
Gradient descent and backpropagation are closely related. Gradient descent is the algorithm used to find the optimal paramaters (weights and biases) for our network, while backpropagation is the process of calculating the gradient that is used in the gradient descent step.

Gradient descent requires some pretty advanced calculus and linear algebra to understand so we'll stay away from that for now. Let's read the formal definition for now.

"Gradient descent is an optimization algorithm used to minimize some function by iteratively moving in the direction of steepest descent as defined by the negative of the gradient. In machine learning, we use gradient descent to update the parameters of our model." (https://ml-cheatsheet.readthedocs.io/en/latest/gradient_descent.html)

Parameters for network are weights and biases,
by changing weight and biases, network becaomes better or worse,
Loss function determines if netowrk is getting better or worse,
helps to determine how to move netowrk to change that.

Gradient Descent image
https://cdn-images-1.medium.com/max/1000/1*iU1QCnSTKrDjIPjSAENLuQ.png
Example of what neural network function might look like


As user has higher dimensional math, there's a lot more spaces to explore when it comes to creating different parameters, biases and activation functions etc.
So when applying Activation Function, network is being spread into higher dimensions which makes stuff much more complex.
What is trying to be done with neural network is optimize this Loss Function.
Loss Function tells how good or bad it is,
If user can get Loss Function as low as possible then user should technically have best Neural Network.
This image is loss function's mapping
But what user is looking for is global minimum,
Minimum point with least possible loss from neural network

If starting where red cirlces are on image, what user should be trying to do is move downward into global minimum.
This is known as process of Gradient Descent.
So calculate loss, use algorithm called Gradient Descent which tells what direction to move function to get to global minimum
Essentially looks where they are, says this was the loss, and calculates a gradient which is literally steepness of a direction and move in that direction.

Next algorithm called backpropogation will go backwards through network and update weights and biases so to move in that direction.


Recap
Neural Networks
Input, output, hidden layers connected with weights, 
there's biases that connect to each layer
biases can be thought of as y intercepts and simply move completely up or completely down that entire activation function.
So shift things left or right because it allows user to get a better prediction and have another parameter that can be trained and add some complexity to that Neural Network Model.

Way information is passed through these layers is to take weighted sum out of neuron of all the neurons connected to it,
Then add bias neuron from Input Layer and apply some Acitvation Function that puts values inbetween two set values.

Example sigmoid squishes values between 0 and 1,
Huperbolic Tangent squishes values between -1 and 1,
Rectified Linear Unit squish values between 0 and positive infinity.
Apply these activation functions and continue process

N1 gets it's value, N2 gets it's value,
Finally make way to Output Layer and might pass through other hidden layers before that,
Then do same thing which is take weighted sum, add bias, apply activation function, look at output and determine whether user knows if it is class y, class z, if that is value looking for and that's how it goes.

This is at the Training Prcoess so this is how this worked when doing a prediction,
So when traning what happens is making predictions, compare those predictions to what expected values should be using this loss function,
Then calculate gradient, gradient is direction needed to move to minimize Loss Function,
Then use algorithm called backpropogation where stepping backwards through network occurs and update weights and biases according to gradient that had been calculated.

More info and data it starts off horrible with no idea of what's going on,
as more info comes in it updates weights and biases and gets better and better as it sees more examples,
After certain amount of epochs or certain amount of pieces of info, network makes better and better predictions and have lower and lower loss.
Way calculating how network is doing is by passing it validation data set where it can say that user got say an 85% accuracy on this data set which is ok,
user can do more changing to improve this

Loss Function better known as Cost Function, lower the better,
is neural networks in a nutshell

Acitvation moves up in dimensionality, 
Bias is anoteher layer of compleity and trainable parameter for network which allows user to shift this kind of activation function left, right, up, down.



