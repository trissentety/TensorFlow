"""
Natural Language Processing with Recurrence Neural Networks
Natural Language Processing (or NLP for short) is a discipline in computing that deals with the communication between natural (human) languages and computer languages. A common example of NLP is something like spellcheck or autocomplete. Essentially NLP is the field that focuses on how computers can understand and/or process natural/human languages.

Recurrent Neural Networks
In this tutorial we will introduce a new kind of neural network that is much more capable of processing sequential data such as text or characters called a recurrent neural network (RNN for short).

We will learn how to use a reccurent neural network to do the following:

Sentiment Analysis
Character Generation
RNN's are complex and come in many different forms so in this tutorial we wil focus on how they work and the kind of problems they are best suited for.


Example Bag of Words
How to turn textual data into numeric data to give to Neural Network
Encoding and preprocess text into integers with Bag of Words which is a popular algorithm for converting textual data to numeric data
Only works right for simple tasks
Observe entire training data set and turn it into form network can understand
Create a dictionary lookup of vocabulary which means every unique word in data set is vocabulary which is amount of words model is expected to understand
Every word in vocabulary is placed in dictionary with some integer that represents it
Example: I,a,Tim,day,Me which goes to length of vocabulary [I,:]
I = 0, a = 1, Tim = 3, day = 3, me = 4
Bag of Words only keeps track of words that are present and frequency of those words
So Bag is created and when word appears, it's number is added into Bag.
If "I am tim day day" is sentence, then [0,1,2,3,3] is what's added to Bag. Order is lost within Bag
In more complex usages different words have different meanings depending on location in sentence, this encoding method becomes flawed
Bag is fed to Neural Network depending on Network used and it tries to do something with integers out of bag
Bag of Words is not used here.


Example 
So when same words have different meanings such as
"I thought the movie was going to be bad, but it was actually amazing!"

"I thought the movie was going to be amazing, but it was actually bad!"

Sentence use same everything but have different meaning in usage.
Meaning of sentence winds up being lost
Bag of words code is written below

If your vocabulary has 100,000 words then that is 100,000 unique mappings from words to integers
1 = happy, 2 = sad, 100000 = good
Grouping words
happy and 100000 = positive. this is an issue with this program
sad = negative
there isn't a a way to group positive and negative words in a integer group close to eachother

Word embedding
Tries to find a way to representent words that are similar using similar numbers
What it does is translate/classify words into a vector. Vector has n amount of dimensions, 64 or 128 for each vector
Every component of vector says what group it belongs to or how similar it is to other words

Word embeddings, embeddings having to do with vectors
Example 3d plane
good instead of integer has vector representation
(x1, x2, x3) Words embedding layer
good and happy would have a similar vector pointing to it in similar angle from eachother
bad would point in different direction and because angle of this word from good is so big they are much different words
Doesn't always work like this in theory but this is what it is trying to do
Word to vector functions by having Word embeddings a a layer added to model which lets it learn embeddings from words
Does this by trying to pick out context in sentence and figure out where word is and what it means and encodes it
Word embeddings are trained and model learns word embeddings as it processes
After it has ran enough training data it has really ood ways of representing all different words so it makes sense to model in further layers
Use pre trained word embeddings layers like pre trained convolusional base in previously





Sequence Data
In the previous tutorials we focused on data that we could represent as one static data point where the notion of time or step was irrelevant. Take for example our image data, it was simply a tensor of shape (width, height, channels). That data doesn't change or care about the notion of time.

In this tutorial we will look at sequences of text and learn how we can encode them in a meaningful way. Unlike images, sequence data such as long chains of text, weather patterns, videos and really anything where the notion of a step or time is relevant needs to be processed and handled in a special way.

But what do I mean by sequences and why is text data a sequence? Well that's a good question. Since textual data contains many words that follow in a very specific and meaningful order, we need to be able to keep track of each word and when it occurs in the data. Simply encoding say an entire paragraph of text into one data point wouldn't give us a very meaningful picture of the data and would be very difficult to do anything with. This is why we treat text as a sequence and process one word at a time. We will keep track of where each of these words appear and use that information to try to understand the meaning of peices of text.

Encoding Text
As we know machine learning models and neural networks don't take raw text data as an input. This means we must somehow encode our textual data to numeric values that our models can understand. There are many different ways of doing this and we will look at a few examples below.

Before we get into the different encoding/preprocessing methods let's understand the information we can get from textual data by looking at the following two movie reviews.

I thought the movie was going to be bad, but it was actually amazing!

I thought the movie was going to be amazing, but it was actually bad!

Although these two setences are very similar we know that they have very different meanings. This is because of the ordering of words, a very important property of textual data.

Now keep that in mind while we consider some different ways of encoding our textual data.

Bag of Words
The first and simplest way to encode our data is to use something called bag of words. This is a pretty easy technique where each word in a sentence is encoded with an integer and thrown into a collection that does not maintain the order of the words but does keep track of the frequency. Have a look at the python function below that encodes a string of text into bag of words.


vocab = {}  # maps word to integer representing it
word_encoding = 1
def bag_of_words(text):
  global word_encoding

  words = text.lower().split(" ")  # create a list of all of the words in the text, well assume there is no grammar in our text for this example
  bag = {}  # stores all of the encodings and their frequency

  for word in words:
    if word in vocab:
      encoding = vocab[word]  # get encoding from vocab
    else:
      vocab[word] = word_encoding
      encoding = word_encoding
      word_encoding += 1
    
    if encoding in bag:
      bag[encoding] += 1
    else:
      bag[encoding] = 1
  
  return bag

text = "this is a test to see if this test will work is is test a a"
bag = bag_of_words(text)
print(bag)
print(vocab)

Run
{1: 2, 2: 3, 3: 3, 4: 3, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1} #Word 1 appears 2 times, Word 2 appears 3 times etc
{'this': 1, 'is': 2, 'a': 3, 'test': 4, 'to': 5, 'see': 6, 'if': 7, 'will': 8, 'work': 9}



This isn't really the way we would do this in practice, but I hope it gives you an idea of how bag of words works. Notice that we've lost the order in which words appear. In fact, let's look at how this encoding works for the two sentences we showed above.


positive_review = "I thought the movie was going to be bad but it was actually amazing"
negative_review = "I thought the movie was going to be amazing but it was actually bad"

pos_bag = bag_of_words(positive_review)
neg_bag = bag_of_words(negative_review)

print("Positive:", pos_bag)
print("Negative:", neg_bag)


We can see that even though these sentences have a very different meaning they are encoded exaclty the same way. Obviously, this isn't going to fly. Let's look at some other methods.

Integer Encoding
The next technique we will look at is called integer encoding. This involves representing each word or character in a sentence as a unique integer and maintaining the order of these words. This should hopefully fix the problem we saw before were we lost the order of words.


vocab = {}  
word_encoding = 1
def one_hot_encoding(text):
  global word_encoding

  words = text.lower().split(" ") 
  encoding = []  

  for word in words:
    if word in vocab:
      code = vocab[word]  
      encoding.append(code) 
    else:
      vocab[word] = word_encoding
      encoding.append(word_encoding)
      word_encoding += 1
  
  return encoding

text = "this is a test to see if this test will work is is test a a"
encoding = one_hot_encoding(text)
print(encoding)
print(vocab)


And now let's have a look at one hot encoding on our movie reviews.


positive_review = "I thought the movie was going to be bad but it was actually amazing"
negative_review = "I thought the movie was going to be amazing but it was actually bad"

pos_encode = one_hot_encoding(positive_review)
neg_encode = one_hot_encoding(negative_review)

print("Positive:", pos_encode)
print("Negative:", neg_encode)


Much better, now we are keeping track of the order of words and we can tell where each occurs. But this still has a few issues with it. Ideally when we encode words, we would like similar words to have similar labels and different words to have very different labels. For example, the words happy and joyful should probably have very similar labels so we can determine that they are similar. While words like horrible and amazing should probably have very different labels. The method we looked at above won't be able to do something like this for us. This could mean that the model will have a very difficult time determing if two words are similar or not which could result in some pretty drastic performace impacts.

Word Embeddings
Luckily there is a third method that is far superior, word embeddings. This method keeps the order of words intact as well as encodes similar words with very similar labels. It attempts to not only encode the frequency and order of words but the meaning of those words in the sentence. It encodes each word as a dense vector that represents its context in the sentence.

Unlike the previous techniques word embeddings are learned by looking at many different training examples. You can add what's called an embedding layer to the beggining of your model and while your model trains your embedding layer will learn the correct embeddings for words. You can also use pretrained embedding layers.

This is the technique we will use for our examples and its implementation will be showed later on.



Recurrent Neural Networks (RNN's)
Now that we've learned a little bit about how we can encode text it's time to dive into recurrent neural networks. Up until this point we have been using something called feed-forward neural networks. This simply means that all our data is fed forwards (all at once) from left to right through the network. This was fine for the problems we considered before but won't work very well for processing text. After all, even we (humans) don't process text all at once. We read word by word from left to right and keep track of the current meaning of the sentence so we can understand the meaning of the next word. Well this is exaclty what a recurrent neural network is designed to do. When we say recurrent neural network all we really mean is a network that contains a loop. A RNN will process one word at a time while maintaining an internal memory of what it's already seen. This will allow it to treat words differently based on their order in a sentence and to slowly build an understanding of the entire input, one word at a time.


difference between rnn and dense neural network or convolusional network is it contains and internal loop.
This means it doesn't process data all at once but at different time steps and maintains an internal memory.
So when it looks at a new input it will remember what it seen previously and treat it based on context of understanding it's already developed.

convolusional and dense neural networks are feed forward neural networks.
So all info is passed through convolusional layer to start then through dense neurons and this information gets translated through network to the end.

Recurrent Neural Network has a loop where it feeds one word at a time so it prcesses word, generates some output based on that word and uses internal memory state that's keeping track of it to do that as part of the calculation and slowly builds up an understanding of what is being read.


This is why we are treating our text data as a sequence! So that we can pass one word at a time to the RNN.

Let's have a look at what a recurrent layer might look like.


https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png
x0 = first word in network
h0 is entire understanding after 1 word. cycle continues making new output. each output cell makes a prediction and feeds to next cell for more processing.
It becomes increastingly difficult for model to have good understanding in general when sequence is long because it's hard for it to remember what it seen at beggining
First layer is Simple RNN
Next layer is LSTM which is Long Short Term Memory. it allows ability to access any prevsious state 


Source: https://colah.github.io/posts/2015-08-Understanding-LSTMs/

Let's define what all these variables stand for before we get into the explination.

ht output at time t

xt input at time t

A Recurrent Layer (loop)

What this diagram is trying to illustrate is that a recurrent layer processes words or input one at a time in a combination with the output from the previous iteration. So, as we progress further in the input sequence, we build a more complex understanding of the text as a whole.

What we've just looked at is called a simple RNN layer. It can be effective at processing shorter sequences of text for simple problems but has many downfalls associated with it. One of them being the fact that as text sequences get longer it gets increasingly difficult for the network to understand the text properly.

LSTM
The layer we dicussed in depth above was called a simpleRNN. However, there does exist some other recurrent layers (layers that contain a loop) that work much better than a simple RNN layer. The one we will talk about here is called LSTM (Long Short-Term Memory). This layer works very similarily to the simpleRNN layer but adds a way to access inputs from any timestep in the past. Whereas in our simple RNN layer input from previous timestamps gradually disappeared as we got further through the input. With a LSTM we have a long-term memory data structure storing all the previously seen inputs as well as when we saw them. This allows for us to access any previous value we want at any point in time. This adds to the complexity of our network and allows it to discover more useful relationships between inputs and when they appear.

For the purpose of this course we will refrain from going any further into the math or details behind how these layers work.

Sentiment Analysis
And now time to see a recurrent neural network in action. For this example, we are going to do something called sentiment analysis.

The formal definition of this term from Wikipedia is as follows:

the process of computationally identifying and categorizing opinions expressed in a piece of text, especially in order to determine whether the writer's attitude towards a particular topic, product, etc. is positive, negative, or neutral.

The example we’ll use here is classifying movie reviews as either postive, negative or neutral.

This guide is based on the following tensorflow tutorial: https://www.tensorflow.org/tutorials/text/text_classification_rnn

Movie Review Dataset
Well start by loading in the IMDB movie review dataset from keras. This dataset contains 25,000 reviews from IMDB where each one is already preprocessed and has a label as either positive or negative. Each review is encoded by integers that represents how common a word is in the entire dataset. For example, a word encoded by the integer 3 means that it is the 3rd most common word in the dataset.

Every word is already encoded by integer and has a integer for how common the word is in data set

%tensorflow_version 2.x  # this line is not required unless you are in a notebook
from keras.datasets import imdb
from keras.preprocessing import sequence
import keras
import tensorflow as tf
import os
import numpy as np

VOCAB_SIZE = 88584 #words in data set with last being least common

MAXLEN = 250
BATCH_SIZE = 64

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = VOCAB_SIZE)
Run
Colab only includes TensorFlow 2.x; %tensorflow_version has no effect.
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb.npz
17464789/17464789 [==============================] - 0s 0us/step




# Lets look at one review
train_data[1]
Run
# Lets look at one review
train_data[1]
Run
[1,
 194,
 1153,
 194,
 8255,
 78,
 228,
 5,
 6,
 1463,
 4369,
 5012,
 134,
 26,
 4,
 715,
 8,
 118,
 1634,
 14,
 394,
 20,
 13,
 119,
 954,
 189,
 102,
 5,
 207,
 110,
 3103,
 21,
 14,
 69,
 188,
 8,
 30,
 23,
 7,
 4,
 249,
 126,
 93,
 4,
 114,
 9,
 2300,
 1523,
 5,
 647,
 4,
 116,
 9,
 35,
 8163,
 4,
 229,
 9,
 340,
 1322,
 4,
 118,
 9,
 4,
 130,
 4901,
 19,
 4,
 1002,
 5,
 89,
 29,
 952,
 46,
 37,
 4,
 455,
 9,
 45,
 43,
 38,
 1543,
 1905,
 398,
 4,
 1649,
 26,
 6853,
 5,
 163,
 11,
 3215,
 10156,
 4,
 1153,
 9,
 194,
 775,
 7,
 8255,
 11596,
 349,
 2637,
 148,
 605,
 15358,
 8003,
 15,
 123,
 125,
 68,
 23141,
 6853,
 15,
 349,
 165,
 4362,
 98,
 5,
 4,
 228,
 9,
 43,
 36893,
 1157,
 15,
 299,
 120,
 5,
 120,
 174,
 11,
 220,
 175,
 136,
 50,
 9,
 4373,
 228,
 8255,
 5,
 25249,
 656,
 245,
 2350,
 5,
 4,
 9837,
 131,
 152,
 491,
 18,
 46151,
 32,
 7464,
 1212,
 14,
 9,
 6,
 371,
 78,
 22,
 625,
 64,
 1382,
 9,
 8,
 168,
 145,
 23,
 4,
 1690,
 15,
 16,
 4,
 1355,
 5,
 28,
 6,
 52,
 154,
 462,
 33,
 89,
 78,
 285,
 16,
 145,
 95] #train_data[1] unique words



More Preprocessing
If we have a look at some of our loaded in reviews, we'll notice that they are different lengths. This is an issue. We cannot pass different length data into our neural network. Therefore, we must make each review the same length. To do this we will follow the procedure below:

if the review is greater than 250 words then trim off the extra words
if the review is less than 250 words add the necessary amount of 0's to make it equal to 250.
Luckily for us keras has a function that can do this for us:


train_data = sequence.pad_sequences(train_data, MAXLEN) #if data is 200 add 50 padding
test_data = sequence.pad_sequences(test_data, MAXLEN)
train_data[1]
array([    0,     0,     0,     0,     0,     0,     0,     0,     0, #padding to make correct length
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     1,   194,
        1153,   194,  8255,    78,   228,     5,     6,  1463,  4369,
        5012,   134,    26,     4,   715,     8,   118,  1634,    14,
         394,    20,    13,   119,   954,   189,   102,     5,   207,
         110,  3103,    21,    14,    69,   188,     8,    30,    23,
           7,     4,   249,   126,    93,     4,   114,     9,  2300,
        1523,     5,   647,     4,   116,     9,    35,  8163,     4,
         229,     9,   340,  1322,     4,   118,     9,     4,   130,
        4901,    19,     4,  1002,     5,    89,    29,   952,    46,
          37,     4,   455,     9,    45,    43,    38,  1543,  1905,
         398,     4,  1649,    26,  6853,     5,   163,    11,  3215,
       10156,     4,  1153,     9,   194,   775,     7,  8255, 11596,
         349,  2637,   148,   605, 15358,  8003,    15,   123,   125,
          68, 23141,  6853,    15,   349,   165,  4362,    98,     5,
           4,   228,     9,    43, 36893,  1157,    15,   299,   120,
           5,   120,   174,    11,   220,   175,   136,    50,     9,
        4373,   228,  8255,     5, 25249,   656,   245,  2350,     5,
           4,  9837,   131,   152,   491,    18, 46151,    32,  7464,
        1212,    14,     9,     6,   371,    78,    22,   625,    64,
        1382,     9,     8,   168,   145,    23,     4,  1690,    15,
          16,     4,  1355,     5,    28,     6,    52,   154,   462,
          33,    89,    78,   285,    16,   145,    95], dtype=int32)


Creating the Model
Now it's time to create the model. We'll use a word embedding layer as the first layer in our model and add a LSTM layer afterwards that feeds into a dense node to get our predicted sentiment.

32 stands for the output dimension of the vectors generated by the embedding layer. We can change this value if we'd like!


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 32), #finds more meaningful representation than only integer values, 32 dimension vectors
    tf.keras.layers.LSTM(32), #lstm needs to know it is 32 as well
    tf.keras.layers.Dense(1, activation="sigmoid") #sigmoid to classify if positive or negative review (0 and 1), final output makes prediction
])


model.summary()
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (None, None, 32)          2834688   #has most numbers to figure out converting to 32 dimensions
                                                                 
 lstm (LSTM)                 (None, 32)                8320      #param changes because of 32
                                                                 
 dense (Dense)               (None, 1)                 33        
                                                                 
=================================================================
Total params: 2843041 (10.85 MB)
Trainable params: 2843041 (10.85 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________

Training
Now it's time to compile and train the model.


model.compile(loss="binary_crossentropy",optimizer="rmsprop",metrics=['acc']) #binary_crossentropy tells how far away from correct probability because 2 things predicting 0 or 1 positive or negative, adam optimizer works too and isn't super important

history = model.fit(train_data, train_labels, epochs=10, validation_split=0.2) #validation_split of 20% of train data to evaluate and validate model as going through
Epoch 1/10
625/625 [==============================] - 48s 72ms/step - loss: 0.4520 - acc: 0.7817 - val_loss: 0.3259 - val_acc: 0.8640
Epoch 2/10
625/625 [==============================] - 20s 33ms/step - loss: 0.2602 - acc: 0.8983 - val_loss: 0.3682 - val_acc: 0.8740
Epoch 3/10
625/625 [==============================] - 12s 19ms/step - loss: 0.1993 - acc: 0.9258 - val_loss: 0.2904 - val_acc: 0.8794
Epoch 4/10
625/625 [==============================] - 11s 17ms/step - loss: 0.1657 - acc: 0.9416 - val_loss: 0.3140 - val_acc: 0.8666
Epoch 5/10
625/625 [==============================] - 10s 15ms/step - loss: 0.1340 - acc: 0.9544 - val_loss: 0.3637 - val_acc: 0.8882
Epoch 6/10
625/625 [==============================] - 10s 15ms/step - loss: 0.1119 - acc: 0.9649 - val_loss: 0.3097 - val_acc: 0.8686
Epoch 7/10
625/625 [==============================] - 9s 15ms/step - loss: 0.0961 - acc: 0.9696 - val_loss: 0.3315 - val_acc: 0.8762
Epoch 8/10
625/625 [==============================] - 9s 14ms/step - loss: 0.0782 - acc: 0.9745 - val_loss: 0.3703 - val_acc: 0.8800
Epoch 9/10
625/625 [==============================] - 9s 14ms/step - loss: 0.0693 - acc: 0.9782 - val_loss: 0.4107 - val_acc: 0.8734
Epoch 10/10
625/625 [==============================] - 8s 13ms/step - loss: 0.0547 - acc: 0.9833 - val_loss: 0.3786 - val_acc: 0.8750
#this tells that there isn't enough training data and something should be changed because stuck on same val_acc and should be better

And we'll evaluate the model on our training data to see how well it performs.


results = model.evaluate(test_data, test_labels)
print(results)
782/782 [==============================] - 5s 6ms/step - loss: 0.4256 - acc: 0.8630
[0.4255642890930176, 0.8629599809646606]
#86% is considered decent because not that much code was written to get to this point

So we're scoring somewhere in the mid-high 80's. Not bad for a simple recurrent network.


Making Predictions
Now let’s use our network to make predictions on our own reviews.

Since our reviews are encoded well need to convert any review that we write into that form so the network can understand it. To do that well load the encodings from the dataset and use them to encode our own data.


#Since data was preprocessed it needs to be processed on anything that is wanted to make a prediction on in same way with same loopup table, encoding otherwise model will think words are different
word_index = imdb.get_word_index() #all word indexes mapped

def encode_text(text): #encodes text into proper preprocessed integers #convert text into tokens which are individual words-
  tokens = keras.preprocessing.text.text_to_word_sequence(text) #for loop
  tokens = [word_index[word] if word in word_index else 0 for word in tokens] #if word in vocabulary then replace it's location in list otherwise put 0 for newc haracter
  return sequence.pad_sequences([tokens], MAXLEN)[0] #pad sequence works on list of sequences, returns list of lists and only returns first entry sequence [0] padded

text = "that movie was just amazing, so amazing"
encoded = encode_text(text)
print(encoded)
Run
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb_word_index.json
1641221/1641221 [==============================] - 0s 0us/step
[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0  12  17  13  40 477  35 477] #integer encoded words with padding


# while were at it lets make a decode function

reverse_word_index = {value: key for (key, value) in word_index.items()}

def decode_integers(integers):
    PAD = 0 #if 0 then nothing is there
    text = "" 
    for num in integers:
      if num != PAD:
        text += reverse_word_index[num] + " " #add lookup number into string

    return text[:-1] #return everything except last space
  
print(decode_integers(encoded))
Run
that movie was just amazing so amazing


# now time to make a prediction

def predict(text):
  encoded_text = encode_text(text) #proper preprocessed text
  pred = np.zeros((1,250)) #blank numpy array with zeros in shape 1,250 because model expects shape 250 integers
  pred[0] = encoded_text #insert entry into array
  result = model.predict(pred) 
  print(result[0]) #result 0 because model is array of arrays and entry is at 0

positive_review = "That movie was! really loved it and would great watch it again because it was amazingly great"
predict(positive_review)

negative_review = "that movie really sucked. I hated it and wouldn't watch it again. Was one of the worst things I've ever watched"
predict(negative_review)
Run
1/1 [==============================] - 0s 356ms/step
[0.8883834] #positivity prediction
1/1 [==============================] - 0s 19ms/step
[0.11612596] #changing sentence changes positivity

RNN Play Generator
Now time for one of the coolest examples we've seen so far. We are going to use a RNN to generate a play. We will simply show the RNN an example of something we want it to recreate and it will learn how to write a version of it on its own. We'll do this using a character predictive model that will take as input a variable length sequence and predict the next character. We can use the model many times in a row with the output from the last predicition as the input for the next call to generate a sequence.

This guide is based on the following: https://www.tensorflow.org/tutorials/text/text_generation


%tensorflow_version 2.x  # this line is not required unless you are in a notebook
from keras.preprocessing import sequence
import keras
import tensorflow as tf
import os
import numpy as np
Run
Colab only includes TensorFlow 2.x; %tensorflow_version has no effect.

Dataset
For this example, we only need one peice of training data. In fact, we can write our own poem or play and pass that to the network for training if we'd like. However, to make things easy we'll use an extract from a shakesphere play.


path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
Run
Downloading data from https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt
1115394/1115394 [==============================] - 0s 0us/step

Loading Your Own Data
To load your own data, you'll need to upload a file from the dialog below. Then you'll need to follow the steps from above but load in this new file instead.


from google.colab import files
path_to_file = list(files.upload().keys())[0]
Run
Upload your own text file and submit

Read Contents of File
Let's look at the contents of the file.


# Read, then decode for py2 compat.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8') #read bytes mode, 
# length of text is the number of characters in it
print ('Length of text: {} characters'.format(len(text)))


# Take a look at the first 250 characters in text
print(text[:250]) 
Run
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.


Encoding
Since this text isn't encoded yet well need to do that ourselves. We are going to encode each unique character as a different integer.


vocab = sorted(set(text)) #sort unique characters in text
# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)} #gives number for eery letter in vocab
idx2char = np.array(vocab) #index to letter mapping rather than lettered inde

def text_to_int(text): #convert text into int representation by referencing character and putting into list
  return np.array([char2idx[c] for c in text])

text_as_int = text_to_int(text) #convert text using funciton


# lets look at how part of our text is encoded
print("Text:", text[:13]) #first 13 letters
print("Encoded:", text_to_int(text[:13]))
Run
Text: First Citizen #each character ha own encoding
Encoded: [18 47 56 57 58  1 15 47 58 47 64 43 52]

def int_to_text(ints): #convert integer back to text
  try:
    ints = ints.numpy() #make numpy array if not already
  except:
    pass
  return ''.join(idx2char[ints]) #if it already is array join all characters

print(int_to_text(text_as_int[:13]))
Run
First Citizen


Creating Training Examples
Remember our task is to feed the model a sequence and have it return to us the next character. This means we need to split our text data from above into many shorter sequences that we can pass to the model as training examples.

The training examples we will prepapre will use a seq_length sequence as input and a seq_length sequence as the output where that sequence is the original sequence shifted one letter to the right. For example:

input: Hell | output: ello

Our first step will be to create a stream of characters from our text data.


seq_length = 100  # length of sequence for a training example input
examples_per_epoch = len(text)//(seq_length+1) #100 characters long output

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int) #convert and split data into characters, stream of 1.1m characters


Next we can use the batch method to turn this stream of characters into batches of desired length.


sequences = char_dataset.batch(seq_length+1, drop_remainder=True) #length is 101 so if 105 characters drop last 4


Now we need to use these sequences of length 101 and split them into input and output.


def split_input_target(chunk):  # for the example: hello
    input_text = chunk[:-1]  # hell
    target_text = chunk[1:]  # ello
    return input_text, target_text  # hell, ello

dataset = sequences.map(split_input_target)  # we use map to apply the above function to every entry, every 101 sequence mapped


for x, y in dataset.take(2):
  print("\n\nEXAMPLE\n")
  print("INPUT")
  print(int_to_text(x))
  print("\nOUTPUT")
  print(int_to_text(y))

Run

0s
for x, y in dataset.take(2):
  print("\n\nEXAMPLE\n")
  print("INPUT")
  print(int_to_text(x))
  print("\nOUTPUT")
  print(int_to_text(y))
output


EXAMPLE

INPUT
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You

OUTPUT
irst Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You 


EXAMPLE

INPUT
are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you 

OUTPUT
re all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:

Finally we need to make training batches.


BATCH_SIZE = 64
VOCAB_SIZE = len(vocab)  # vocab is number of unique characters, sorted text with unique characters
EMBEDDING_DIM = 256
RNN_UNITS = 1024

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True) #switch around sequences, 64 batches is 64 training examples at a time, drop remainder if not enough batches


Building the Model
Now it is time to build the model. We will use an embedding layer a LSTM and one dense layer that contains a node for each unique character in our training data. The dense layer will give us a probability distribution over all nodes.

#return 64 batches
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]), #none cause not sure how long sequence is, 1 entry
    tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True, #otherwise returns only 1 output per time step
                        stateful=True,
                        recurrent_initializer='glorot_uniform'), #default from TensorFlow
    tf.keras.layers.Dense(vocab_size) #amount of vocab size nodes, final layer have same amount of nodes equal to amount of chars in vocab so all node values sum together give value of 1
  ])
  return model

model = build_model(VOCAB_SIZE,EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
model.summary()
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (64, None, 256)           16640      #64 is batch size, none is length of sequence, amount of values in vector
                                                                 
 lstm (LSTM)                 (64, None, 1024)          5246976   
                                                                 
 dense (Dense)               (64, None, 65)            66625     
                                                                 
=================================================================
Total params: 5330241 (20.33 MB)
Trainable params: 5330241 (20.33 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________

Creating a Loss Function
Now we are going to create our own loss function for this problem. This is because our model will output a (64, sequence_length, 65) shaped tensor that represents the probability distribution of each character at each timestep for every sequence in the batch.

However, before we do that let's have a look at a sample input and the output from our untrained model. This is so we can understand what the model is giving us.

#64 entries all of length 100 into model as training data

for input_example_batch, target_example_batch in data.take(1):
  example_batch_predictions = model(input_example_batch)  # ask our model for a prediction on our first batch of training data (64 entries)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")  # print out the output shape
Run
(64, 100, 65) # (batch_size, sequence_length, vocab_size)

# we can see that the predicition is an array of 64 arrays, one for each entry in the batch
print(len(example_batch_predictions))
print(example_batch_predictions)
Run
[[-8.96180398e-04  4.66918480e-03 -4.38368297e-05 ... -4.12193872e-03 #list inside list with different predictions
   -7.15726614e-03 -7.54947821e-03]
  [-1.58750033e-03  1.23622059e-03 -2.92591285e-05 ... -4.14183317e-03
   -5.97598497e-03 -6.52788486e-03]
  [-7.60016497e-03  2.65959231e-03 -1.90272729e-03 ... -1.87689916e-03
   -6.48217509e-03 -7.19526876e-03]
  ...
  [ 1.40483608e-03  1.41561334e-03 -5.23759471e-03 ...  1.89767638e-03
   -1.24127828e-02 -1.30567327e-02]
  [-3.00825899e-03  6.39455905e-03 -4.78126574e-04 ...  6.11833064e-03
   -1.88318696e-02 -6.35566283e-03]
  [-2.99863727e-03  9.46004689e-03 -1.60779629e-03 ...  8.69406853e-04
   -2.25139689e-02 -1.19049232e-02]]

 ...

 [[ 2.77939904e-03 -4.09891317e-03 -1.11406180e-03 ... -2.01103720e-03
   -9.10484989e-04  1.05251896e-03]
  [ 3.14097246e-03 -3.22878221e-03 -6.19881088e-04 ... -1.19409477e-03
   -2.39307946e-03 -2.55956757e-03]
  [ 9.66345053e-03 -2.60066194e-03 -1.70541205e-03 ... -2.35862029e-03
   -2.97307875e-03 -4.13512578e-03]
  ...
  [ 6.54114643e-03  2.52312608e-03 -1.29006645e-02 ...  3.53840715e-03
   -8.14745668e-04  3.51945963e-03]
  [ 1.05961757e-02 -2.27089366e-03 -9.46685579e-03 ... -2.46623810e-03
    2.05111457e-03  1.27493986e-05]
  [ 7.62097910e-03  3.16276914e-03 -7.89050199e-03 ... -7.65502639e-03
   -4.19072900e-03 -8.13997816e-03]]

 [[ 5.93113434e-03 -3.71965603e-03  2.08724698e-04 ... -3.49016185e-03
    1.93193555e-05 -1.54605135e-03]
  [ 7.40151759e-03 -5.78228803e-03 -9.86291561e-04 ... -5.05235046e-03
   -1.21786515e-03  2.93595367e-04]
  [ 5.50074677e-04  2.45000701e-03  4.29225247e-03 ...  2.42071692e-04
   -9.73160658e-03  3.03345500e-03]
  ...
  [ 5.58310747e-03  9.22752544e-03  5.09722577e-03 ... -5.86359156e-03
   -1.26971267e-02 -4.76302719e-03]
  [ 1.01059247e-02  7.32868165e-03  1.27784593e-03 ... -8.73496849e-03
   -1.02535402e-02 -6.70908485e-03]
  [ 6.52803760e-03  9.98218358e-03  4.57186252e-05 ... -1.26680983e-02
   -1.36874039e-02 -1.21172778e-02]]

 [[-1.77532830e-03 -3.89185268e-03 -1.80862774e-03 ...  5.00902394e-03
    6.21921569e-03  1.68448570e-03]
  [-2.47465982e-03  1.61134626e-03 -6.96362928e-04 ... -3.35107790e-04
   -3.09378281e-03 -5.70769468e-03]
  [ 3.32118082e-03 -4.51820344e-03 -7.48957740e-04 ... -1.96507550e-04
   -4.69424250e-03 -8.39342922e-03]
  ...
  [ 1.14130946e-02 -8.92495550e-03 -5.64192468e-03 ... -3.16146039e-03
   -7.88502954e-03 -1.16304839e-02]
  [ 2.64505041e-03 -5.44106867e-03 -6.83087483e-03 ... -1.35514489e-03
   -8.43678787e-03 -1.06509952e-02]
  [ 9.73176304e-03 -9.49054025e-03 -6.00979710e-03 ... -4.59826738e-03
   -7.23758526e-03 -9.06744972e-03]]], shape=(64, 100, 65), dtype=float32)

# lets examine one prediction
pred = example_batch_predictions[0]
print(len(pred))
print(pred)
# notice this is a 2d array of length 100, where each interior array is the prediction for the next character at each time step
Run
100
tf.Tensor(
[[ 0.0027794  -0.00409891 -0.00111406 ... -0.00201104 -0.00091048 #predictions
   0.00105252]
 [-0.0007207  -0.00041139  0.00387191 ... -0.00533567 -0.00763047
   0.0027708 ]
 [-0.00128329  0.00425681  0.00374277 ... -0.00918228 -0.01266128
  -0.00500956]
 ...
 [ 0.00594407 -0.00254304 -0.00592383 ... -0.01783788 -0.00717588
  -0.01318505]
 [ 0.00356345  0.00380866 -0.00478552 ... -0.01964829 -0.01305964
  -0.01775493]
 [ 0.00420873  0.00273955 -0.00462802 ... -0.01674588 -0.01260488
  -0.01738268]], shape=(100, 65), dtype=float32)

# and finally well look at a prediction at the first timestep
time_pred = pred[0]
print(len(time_pred))
print(time_pred)
# and of course its 65 values representing the probabillity of each character occuring next
65
tf.Tensor(
[ 2.7793990e-03 -4.0989132e-03 -1.1140618e-03  1.0927886e-03 #1 prediction, probability of every single character occurring next at the first time step
 -3.4160363e-03  5.9579290e-03  6.7019514e-03 -3.3583415e-03
  5.6364192e-03 -9.6647278e-04 -5.9812353e-04 -9.0201304e-04
  6.3953339e-05  1.5137885e-03 -5.7252850e-03 -1.7182718e-03
 -1.1130503e-03 -2.2362424e-03  2.0675901e-03  2.4572890e-03
 -5.4885428e-03  4.1763363e-03  2.0554638e-05  2.4646688e-03
  4.9838247e-03  1.3308664e-03 -4.7911471e-03  1.8992126e-03
  5.1680258e-03 -2.2365199e-03  1.6362421e-03 -5.8900346e-03
 -2.9800492e-03  2.4824329e-03 -5.3210184e-04  4.3047406e-03
  2.7754023e-03 -3.4508859e-03 -5.4791276e-03 -4.9652630e-03
  9.1557857e-04  1.8158843e-03 -3.1668006e-03 -3.8917284e-03
 -8.9159980e-03 -1.4142598e-03  4.9576159e-03 -9.9073420e-04
 -3.3628924e-03 -2.4448140e-03 -5.0879875e-03 -3.2605687e-03
  1.6788589e-03  2.3687058e-03 -1.3251344e-03  1.0428810e-03
  8.5565238e-04  2.8386135e-03 -5.1059091e-04 -3.7803524e-03
  3.2780180e-04 -6.9321529e-04 -2.0110372e-03 -9.1048499e-04
  1.0525190e-03], shape=(65,), dtype=float32)


# If we want to determine the predicted character we need to sample the output distribution (pick a value based on probabillity)
#sampling probability distribution rather than picking highest values
sampled_indices = tf.random.categorical(pred, num_samples=1)
Run
obIbt qYUu-iGO&TVRPJauTRmCKSJ'Uk&Zm$XdajDu:tHxHL. L- EXZgIU-.oBxrT-Drk HKKgJYKmMIhwD:fXc.3MztAQ\n3SKP #all characters predicted. prediction at time step 0 to next character

# now we can reshape that array and convert all the integers to numbers to see the actual characters
sampled_indices = np.reshape(sampled_indices, (1, -1))[0] #reshape array and convert integers to numbers
predicted_chars = int_to_text(sampled_indices)

predicted_chars  # and this is what the model predicted for training sequence 1


So now we need to create a loss function that can compare that output to the expected output and give us some numeric value representing how close the two were.


def loss(labels, logits): #keras built in loss function, takes labels and probability distribution and computes loss which is how different or similar two things are. goal in algorithm is to reduce loss
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


Compiling the Model
At this point we can think of our problem as a classification problem where the model predicts the probabillity of each unique letter coming next.


model.compile(optimizer='adam', loss=loss)


Creating Checkpoints
Now we are going to setup and configure our model to save checkpoinst as it trains. This will allow us to load our model from a checkpoint and continue training it.


# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)


Training
Finally, we will start training the model.

If this is taking a while go to Runtime > Change Runtime Type and choose "GPU" under hardware accelerator.


history = model.fit(data, epochs=50, callbacks=[checkpoint_callback]) #save each epoch 
Epoch 1/50
172/172 [==============================] - 18s 69ms/step - loss: 2.5611
Epoch 2/50
172/172 [==============================] - 14s 66ms/step - loss: 1.8753
Epoch 3/50
172/172 [==============================] - 13s 67ms/step - loss: 1.6324
Epoch 4/50
172/172 [==============================] - 13s 67ms/step - loss: 1.5026
Epoch 5/50
172/172 [==============================] - 16s 69ms/step - loss: 1.4229
Epoch 6/50
172/172 [==============================] - 14s 69ms/step - loss: 1.3671
Epoch 7/50
172/172 [==============================] - 14s 70ms/step - loss: 1.3226
Epoch 8/50
172/172 [==============================] - 14s 71ms/step - loss: 1.2834
Epoch 9/50
172/172 [==============================] - 14s 72ms/step - loss: 1.2475
Epoch 10/50
172/172 [==============================] - 15s 72ms/step - loss: 1.2122
Epoch 11/50
172/172 [==============================] - 15s 73ms/step - loss: 1.1771
Epoch 12/50
172/172 [==============================] - 14s 73ms/step - loss: 1.1411
Epoch 13/50
172/172 [==============================] - 14s 73ms/step - loss: 1.1043
Epoch 14/50
172/172 [==============================] - 14s 72ms/step - loss: 1.0644
Epoch 15/50
172/172 [==============================] - 14s 72ms/step - loss: 1.0252
Epoch 16/50
172/172 [==============================] - 15s 71ms/step - loss: 0.9834
Epoch 17/50
172/172 [==============================] - 15s 75ms/step - loss: 0.9424
Epoch 18/50
172/172 [==============================] - 14s 74ms/step - loss: 0.9014
Epoch 19/50
172/172 [==============================] - 15s 74ms/step - loss: 0.8619
Epoch 20/50
172/172 [==============================] - 14s 72ms/step - loss: 0.8236
Epoch 21/50
172/172 [==============================] - 15s 73ms/step - loss: 0.7865
Epoch 22/50
172/172 [==============================] - 14s 74ms/step - loss: 0.7514
Epoch 23/50
172/172 [==============================] - 14s 72ms/step - loss: 0.7213
Epoch 24/50
172/172 [==============================] - 14s 74ms/step - loss: 0.6898
Epoch 25/50
172/172 [==============================] - 15s 72ms/step - loss: 0.6636
Epoch 26/50
172/172 [==============================] - 15s 75ms/step - loss: 0.6394
Epoch 27/50
172/172 [==============================] - 14s 74ms/step - loss: 0.6167
Epoch 28/50
172/172 [==============================] - 14s 73ms/step - loss: 0.5981
Epoch 29/50
172/172 [==============================] - 14s 72ms/step - loss: 0.5810
Epoch 30/50
172/172 [==============================] - 14s 72ms/step - loss: 0.5644
Epoch 31/50
172/172 [==============================] - 15s 75ms/step - loss: 0.5491
Epoch 32/50
172/172 [==============================] - 15s 73ms/step - loss: 0.5361
Epoch 33/50
172/172 [==============================] - 14s 73ms/step - loss: 0.5250
Epoch 34/50
172/172 [==============================] - 14s 74ms/step - loss: 0.5129
Epoch 35/50
172/172 [==============================] - 14s 73ms/step - loss: 0.5043
Epoch 36/50
172/172 [==============================] - 14s 72ms/step - loss: 0.4959
Epoch 37/50
172/172 [==============================] - 16s 74ms/step - loss: 0.4867
Epoch 38/50
172/172 [==============================] - 15s 77ms/step - loss: 0.4798
Epoch 39/50
172/172 [==============================] - 15s 74ms/step - loss: 0.4726
Epoch 40/50
172/172 [==============================] - 14s 73ms/step - loss: 0.4670
Epoch 41/50
172/172 [==============================] - 14s 72ms/step - loss: 0.4616
Epoch 42/50
172/172 [==============================] - 14s 71ms/step - loss: 0.4591
Epoch 43/50
172/172 [==============================] - 15s 74ms/step - loss: 0.4541
Epoch 44/50
172/172 [==============================] - 16s 76ms/step - loss: 0.4492
Epoch 45/50
172/172 [==============================] - 14s 74ms/step - loss: 0.4451
Epoch 46/50
172/172 [==============================] - 14s 72ms/step - loss: 0.4410
Epoch 47/50
172/172 [==============================] - 14s 72ms/step - loss: 0.4389
Epoch 48/50
172/172 [==============================] - 14s 72ms/step - loss: 0.4349
Epoch 49/50
172/172 [==============================] - 14s 72ms/step - loss: 0.4324
Epoch 50/50
172/172 [==============================] - 15s 75ms/step - loss: 0.4296

Loading the Model
We'll rebuild the model from a checkpoint using a batch_size of 1 so that we can feed one peice of text to the model and have it make a prediction.


model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, batch_size=1) #rebuild using new batch size of 1x`c`x`


Once the model is finished training, we can find the lastest checkpoint that stores the models weights using the following line.


model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))


We can load any checkpoint we want by specifying the exact file to load.


checkpoint_num = 10
model.load_weights(tf.train.load_checkpoint("./training_checkpoints/ckpt_" + str(checkpoint_num)))
model.build(tf.TensorShape([1, None]))


Generating Text
Now we can use the lovely function provided by tensorflow to generate some text using any starting string we'd like.
Provided by TensorFlow

def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = 800

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string] #start_string is value typed in
  input_eval = tf.expand_dims(input_eval, 0)#expecting double list [[]] as output

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0 

  # Here batch size == 1
  model.reset_states() #because it stores last state remembers training
  for i in range(num_generate):
      predictions = model(input_eval) #start string encoded
      # remove the batch dimension
    
      predictions = tf.squeeze(predictions, 0) #takes predicions in nested list and removes exterior dimension

      # using a categorical distribution to predict the character returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy() #sample output from model

      # We pass the predicted character as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0) #add predicted id to input eval

      text_generated.append(idx2char[predicted_id]) #convert text of integers back into string

  return (start_string + ''.join(text_generated))


inp = input("Type a starting string: ")
print(generate_text(model, inp))


And that's pretty much it for this module! I highly reccomend messing with the model we just created and seeing what you can get it to do!

Sources
Chollet François. Deep Learning with Python. Manning Publications Co., 2018.
“Text Classification with an RNN  :   TensorFlow Core.” TensorFlow, www.tensorflow.org/tutorials/text/text_classification_rnn.
“Text Generation with an RNN  :   TensorFlow Core.” TensorFlow, www.tensorflow.org/tutorials/text/text_generation.
“Understanding LSTM Networks.” Understanding LSTM Networks -- Colah's Blog, https://colah.github.io/posts/2015-08-Understanding-LSTMs/.



