### Report

what is a convolutional neural network made up of?
A CNN is made up of {what fundamental parts}.

#### what can be used for this project
In this project we are given the task to replicate a paper's results and code can be taken from labs as a baseline. But the rest of the code should be orginally yours. We are able to use the Tensorflow library for building the neural network. But we are not allowed other software engines. So I can assume we can use other libraries as long it is not a deep learning library to implement the architecture.

##### The structure of the report has been given
* report in IEEE conference
* upto 5 pages including references
* Title
* intro
* related work
* dataset
* Method
* Implementation details
* replication figures
* replicating quantitative results
* discussion
* improvements
* conclusion and future work
[more detail is given in the task paper](https://www.ole.bris.ac.uk/bbcswebdav/courses/COMSM0018_2017/content/COMSM0018_Project_final.pdf)
---

#### convolutional layer
**convolution basis**
In image recognition special kinds of kernels (matrices) were applied to images through convolution, it is a special way of applying a function over the image(pixels). The purpose was to extract specific features depending on how these kernels were designed (aka operators)

Convolutional layer is this, but is called a layer because it is a small bit of computation made over the entire process of the network.


---

#### notes (to not include in the report) Tensorflow
There are two core features to make tensorflow work. 
1. Build the computational graph
2. Run the computational graph

A computational graph is a series of Tensorflow operations arranged in graph of nodes.
you first set up the values, but they are not built until you run ```sess.run()```

a node a tensorflow object that can be many things like constants and functions from the perspective of ```sess.run()```. The idea is similar to a python objects to the python interpreter.

To be able to keep track of these nodes Tensorflow provides a visualisation of the computational graph with the **Tensorboard**.

##### placeholder
we can give graphs external inputs (parameters) known as placeholders. It is a promise to provide the value later.

#### tensorflow variables
Why: To be able to modify the graph to get new outputs with the same input. Variables allows us to trainable parameters to a graph.
```tf.constants()``` do not need to be initialised they were able to pass into the session directly. However, variables must be initialised first.
```init == tf.global_variables_intializer()``` and we can pass init into sess.

what doe this mean?
* It is important to realize init is a handle to the TensorFlow sub-graph that initializes all the global variables. Until we call sess.run, the variables are uninitialized.

#### loss function
A loss function measures how far apart the current model is from the the provided data.
We will use a standard loss model for linear regression. That is to measure the distance from the current model our (data plots forming the line) is from the actual data.

#### weights & bias
The weights and bias are parameters to be tuned in order to reduce the loss function value. We are moving the curve to fit the data better.
Finding the perfect weights and bias is a real challenge. To find these values we will most likely be performing some sort of statistical method (aka machine learnign).

#### minising the loss function (optimisation)
Tensorflow provides **optimisers** that slowly change each variable in order to minimise the loss function.
The simplest optimiser is known as gradient decent.
It modifies each variable according to the magnitude of the derivate of the loss w.r.t to the variable.
We are optimising loss function against the variable to find the minimum (like in a levels maths)
