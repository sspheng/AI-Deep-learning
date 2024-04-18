A. WHAT IS ARTIFICIAL NEURAL NETWORK(ANN) ?

The term "Artificial neural network" refers to a biologically inspired sub-field of artificial intelligence modeled after the brain. 

An Artificial neural network is usually a computational network based on biological neural networks that construct the structure of the human brain. 

Similar to a human brain has neurons interconnected to each other, artificial neural networks also have neurons that are linked to each other in various layers of the networks. These neurons are known as nodes.

** Artificial Neural Network primarily consists of three layers:

1. Input Layer: As the name suggests, it accepts inputs in several different formats provided by the programmer.

2. Hidden Layer: The hidden layer presents in-between input and output layers. It performs all the calculations to find hidden features and patterns.

3.Output Layer:The input goes through a series of transformations using the hidden layer, which finally results in output that is conveyed using this layer. The artificial neural network takes input and computes the weighted sum of the inputs and includes a bias. This computation is represented in the form of a transfer function.

A1. WHAT ARE THE TYPES OF ANN ?

There are various types of Artificial Neural Networks (ANN) depending upon the human brain neuron and network functions, an artificial neural network similarly performs tasks. The majority of the artificial neural networks will have some similarities with a more complex biological partner and are very effective at their expected tasks. 
For example, segmentation or classification.

a). Feedback ANN:
In this type of ANN, the output returns into the network to accomplish the best-evolved results internally. The feedback networks feed information back into itself and are well suited to solve optimization issues. The Internal system error corrections utilize feedback ANNs.

b). Feed-Forward ANN:
A feed-forward network is a basic neural network comprising of an input layer, an output layer, and at least one layer of a neuron. Through assessment of its output by reviewing its input, the intensity of the network can be noticed based on group behavior of the associated neurons, and the output is decided. 
The primary advantage of this network is that it figures out how to evaluate and recognize input patterns.

A2. WHAT ARE THE TOOLS FOR DEEP LEARNING ?

1. Torch:

The torch deep learning tools is an exceptionally effective open-source program. This logical figuring system is supporting ML algorithms utilizing a Graphics Processing Unit. It utilizes a powerful LuaJIT scripting language and a basic Compute Unified Device Architecture execution. The light has a transposing, slicing, lots of routines for indexing, powerful N-dimensional array feature, and so on. 
Has fantastic Graphics Processing Unit support and is embeddable so it can work with Android, iOS, and so on

2. Neural Designer: 

Neural Designer is an expert application to find hidden designs, complicated connections, and anticipating genuine patterns from data indexes utilizing neural networks. The Spain based new business Artelnics created Neural Designer, which has gotten quite possibly the most mainstream desktop applications for data mining. 
It utilizes neural networks as numerical models imitating the human cerebrum work. Neural Designer constructs computational models that work as the focal sensory system.

3. TensorFlow: 

Deep Learning Tools TensorFlow is regularly utilized across an assortment of undertakings yet includes a specific spend significant time in inference and training of deep neural networks. It might be a representative mathematical library that upheld differentiable and dataflow programming. 
It encourages the structure of both factual Machine Learning or ML arrangements just as profound deep learning through its broad interface of Compute Unified Device Architecture and Graphics Processing Unit. It Offers help and capacities for different utilizations of Machine Learning like Reinforcement Learning, Natural Language Processing and Computer Vision. 
TensorFlow is one of the must-know devices of ML for newcomers.

4. Microsoft Cognitive Toolkit: 

Microsoft Cognitive Toolkit deep learning tools is a monetarily utilized tool compartment that trains deep learning frameworks to adapt exactly like a human mind.It is effortless and free open-source to utilize. It furnishes excellent scaling abilities alongside enterprise-level quality, accuracy and speed. 
It enables clients to bridle the knowledge inside huge datasets through data learning. Microsoft Cognitive Toolkit deep learning tools depicts neural networks as an arrangement of computational strides through a coordinated diagram.

5. Pytorch: 

Pytorch is a deep learning tool. It is exceptionally quick just as adaptable to utilize. This is because Pytorch has a decent order over the Graphics Processing Unit. It is quite possibly the main apparatuses of ML since it is utilized in the most indispensable parts of machine learning which incorporate constructing tensor calculations and deep neural networks. 
Pytorch deep learning tool is founded on Python. Alongside this, it is the best option in contrast to NumPy.

6. H20.ai: 

H20’s deep learning tool gives a versatile multi-layer AI neural network. H20’s possibly an entirely open-source, appropriated in-memory ML stage with direct adaptability. H20’s backings the principal broadly utilized measurable and ML calculations including deep learning, generalized linear models, gradient boosted machines, and so on. 
This artificial neural network includes a few parameters and components which can be modified likewise dependent on the data stored. It likewise contains a rate-adaptive and annealing learning rate to output profoundly prescient yield.

7. Keras: 

Keras deep learning tool is a deep learning library that has negligible functionalities. Keras deep learning tool was created with attention to empowering quick experimentation and works with TensorFlow and Theano. The key advantage is that it can take you from thought to bring about a quick speed. 
Keras deep learning tool is created in Python and fills in as an undeniable level neural networks library fit for running on top of either Theano or TensorFlow. It takes into consideration simple and quick prototyping utilizing minimalism, extensibility, and total modularity. 
Keras deep learning tool underpins recurrent networks, convolutional networks, a combination of both, and self-assertive availability plans like multi-output and multi-input training.

**WHAT IS TUNING ?

The hyperparameters to tune are the number of neurons, activation function, optimizer, learning rate, batch size, and epochs. The second step is to tune the number of layers.

**WHAT IS AUTO ENCODERS?

Autoencoders are a type of neural network that learns the data encodings from the dataset in an unsupervised way. It basically contains two parts: the first one is an encoder which is similar to the convolution neural network except for the last layer. The aim of the encoder to learn efficient data encoding from the dataset and pass it into a bottleneck architecture.
The other part of the Autoencoder is a decoder that uses latent space in the bottleneck layer to regenerate the images similar to the dataset. These results backpropagate from the neural network in the form of the loss function.

B. WHAT IS CONVOLUTION ?

A Convolutional Neural Network (ConvNet/CNN) is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other. The pre-processing required in a ConvNet is much lower as compared to other classification algorithms.
While in primitive methods filters are hand-engineered, with enough training, ConvNets have the ability to learn these filters/characteristics.

FORWARD PROPAGATION

Forward propagation (or forward pass) refers to the calculation and storage of intermediate variables (including outputs) for a neural network in order from the input layer to the output layer. We now work step-by-step through the mechanics of a neural network with one hidden layer.
This may seem tedious but in the eternal words of funk virtuoso James Brown, you must “pay the cost to be the boss”. For the sake of simplicity, let us assume that the input example is  x∈Rd  and that our hidden layer does not include a bias term. Here the intermediate variable is:
z=W(1)x,

BACKPROPAGATION FOR CNNS

Backpropagation refers to the method of calculating the gradient of neural network parameters.In short, the method traverses the network in reverse order, from the output to the input layer, according to the chain rule from calculus. The algorithm stores any intermediate variables (partial derivatives) required while calculating the gradient with respect to some parameters.
Assume that we have functions  Y=f(X)  and  Z=g(Y) , in which the input and the output  X,Y,Z  are tensors of arbitrary shapes. By using the chain rule, we can compute the derivative of  Z  with respect to  X  via
 ∂Z∂X=prod(∂Z∂Y,∂Y∂X).

*** WHAT IS CNN ARCHITECTURE ?

The architecture of CNN is the most important factor that analyses the performance and determines accuracy. An arrangement of layers in the network and the filters used in each layer impacts the performance of an algorithm a lot. Through convolutional neural networks, machines can visualize the real world like humans and CNN will always be the go-to model to recognize the objects.

There are two main parts to a CNN architecture

A convolution tool that separates and identifies the various features of the image for analysis in a process called as Feature Extraction

A fully connected layer that utilizes the output from the convolution process and predicts the class of the image based on the features extracted in previous stages


1. Convolutional Layer
   
This layer is the first layer that is used to extract the various features from the input images. In this layer, the mathematical operation of convolution is performed between the input image and a filter of a particular size MxM. By sliding the filter over the input image, the dot product is taken between the filter and the parts of the input image with respect to the size of the filter (MxM).
The output is termed as the Feature map which gives us information about the image such as the corners and edges. Later, this feature map is fed to other layers to learn several other features of the input image.

2. Pooling Layer
   
In most cases, a Convolutional Layer is followed by a Pooling Layer. The primary aim of this layer is to decrease the size of the convolved feature map to reduce the computational costs. This is performed by decreasing the connections between layers and independently operates on each feature map. 
Depending upon method used, there are several types of Pooling operations.In Max Pooling, the largest element is taken from feature map. Average Pooling calculates the average of the elements in a predefined sized Image section. 
The total sum of the elements in the predefined section is computed in Sum Pooling. The Pooling Layer usually serves as a bridge between the Convolutional Layer and the FC Layer

3. Fully Connected Layer

The Fully Connected (FC) layer consists of the weights and biases along with the neurons and is used to connect the neurons between two different layers. These layers are usually placed before the output layer and form the last few layers of a CNN Architecture.In this, the input image from the previous layers are flattened and fed to the FC layer. The flattened vector then undergoes few more FC layers where the mathematical functions operations usually take place. 
In this stage, the classification process begins to take place.

4. Dropout

Usually, when all the features are connected to the FC layer, it can cause overfitting in the training dataset. Overfitting occurs when a particular model works so well on the training data causing a negative impact in the model’s performance when used on a new data.To overcome this problem, a dropout layer is utilised wherein a few neurons are dropped from the neural network during training process resulting in reduced size of the model. 
On passing a dropout of 0.3, 30% of the nodes are dropped out randomly from the neural network.

5. Activation Functions

Finally, one of the most important parameters of the CNN model is the activation function. They are used to learn and approximate any kind of continuous and complex relationship between variables of the network. In simple words, it decides which information of the model should fire in the forward direction and which ones should not at the end of the network.It adds non-linearity to the network. 
There are several commonly used activation functions such as the ReLU, Softmax, tanH and the Sigmoid functions. Each of these functions have a specific usage.For a binary classification CNN model, sigmoid and softmax functions are preferred an for a multi-class classification, generally softmax us used.

C. WHAT IS RECURRENT NEURAL NETWORK?

Recurrent Neural Network(RNN) are a type of Neural Network where the output from previous step are fed as input to the current step. In traditional neural networks, all the inputs and outputs are independent of each other, but in cases like when it is required to predict the next word of a sentence, the previous words are required and hence there is a need to remember the previous words. 
Thus RNN came into existence, which solved this issue with the help of a Hidden Layer. The main and most important feature of RNN is Hidden state, which remembers some information about a sequence.

There are four types of Recurrent Neural Networks:

One to One: This type of neural network is known as the Vanilla Neural Network. It's used for general machine learning problems, which has a single input and a single output.

One to Many: This type of neural network has a single input and multiple outputs. An example of this is the image caption.

Many to One: This RNN takes a sequence of inputs and generates a single output. Sentiment analysis is a good example of this kind of network where a given sentence can be classified as expressing positive or negative sentiments.

Many to Many: This RNN takes a sequence of inputs and generates a sequence of outputs. Machine translation is one of the examples.

Reference: 

https://my.careerera.com/














