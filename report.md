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

#### with block with scopes
you might be wondering what are with these with blocks.
It is tensorflow way of sharing variables. Similar to scope in terms of functions and the global. Tensorflow has a similar concept with graphs. The with blocks allow tensorflow to share the variables.

```python
with tf.variable_scope('scope_name'):
    tf.Variable(..)
    tf.get_variable('b', ...) # <-- this allows retrieve the variable from a certain scope
```

#### logits
in maths in a particular function

The name appears in the program. For tensorflow: it\s a name that is though to imply that this Tensori is the quantity that is being mapped to probabilities by Softmax. Logics are values to be used as input into softmax.

* **Logit** is a function that maps probabilities [0,1] to [-inf, inf]
* **Softmax** is a function that maps [-inf, inf] to [0,1] similar to a Sigmoid. But Softmax also normliases the sum of the values (output vector) to be 1.
* **Tensorflow "with logit"**: Means you are applying a softmax function to logit numers to normalise it. The input_vector (logit) is not normalised and is in the range [-inf,inf]. 

This normalisation is usd for multiclass classification problems. And for multilabel classification problems sigmoid normisation is used i.e
```python
tf.nn.sigmoid_cross_entropy_with_logits
```

---
### basics of tensorflow

**computational graph**
Tensorflow uses a series of computations as a flow of data through a graph.
* nodes being computation units
* edges being the flow of Tensors (multidimensional arrays)
* Tensorflow builds the computation graph before it starts execution (it follows the principles of lazy programming - computation when absolutely necessary) 
* The graph is not executed when the nodes are defined. After the graph is assembled, it is deplyoyed and executed in a **Session**.
* The *session* is the run time environment we are familiar with a typical execution of a python program.
* **Tensorboard** is very useful to visualise and debug since we are able to see every node (computation unit) and Tensor (edge)

**Tensorflow is a low level library**
What I mean by low level is that it comes with features that allows us to build architectures, rather than supplying the architectures themselves.
* Tensorflow is like numpy (operations) rather than scikit-learn (a toolbox of algorithms that has already been created by *numpy*)
* Tensorflow provides us with some very simple operators (very much like numpy) but was build to efficiently deal with Tensors

**Auto differentiation**
* A powerful tool used frequently in deep learning.
* It gives the user the ability to automatically calculate derivatives
* Tensorflow efficiently calculates derivatives from the computation graph by using chain rule, where every node node has an attached gradient operation which calculates derivatives of input w.r.t. the output.
* Gradients w.r.t. the parameters are calculated automatically during backpropagation.

### Understanding the computation graph
Essentially, Tensorflow computation graph contains the following parts:
**ingredients for our computational graph (before execution)**
1. **placeholders**, variables used in place of inputs to feed to the graph
2. **Variables**, model variables that are going to be *optimised* to make the model perform better
3. **Model**, a mathematical function that calculates a output based on the placeholder and model variables. i.e. Model(placeholder, Variables) -> someoutput.
4. **Loss Measure**, guide for optimisation for model variables (the criteria in which the model should be minimising aka. minimising the loss function)
5. **Optimisation Method**, update method for tuning model variables, (how should we update the Variables during the tuning (i.e. during the minimising loss values)

### things to note
* while defining the graph i.e. coding the architecture. The graph is not executed which means there is no data manipulation. Only building the nodes and symbols inside our graph.
* we do not create the graph strucutre explicitly in our programs. New nodes are automatially build into the underlying graph. we can use
```python
tf.get_default_graph().get_operations() # see all the nodes in default graph
# we can probably get more info in the api for
tf.get_default_graph()
```

### more about Session
After building your computational graph. The next thing to do is to execute it.
* Session is a binding to a particular execution enviroment (CPU or GPU). A session object creates a runtime where operation nodes are executed and tensors are evaluated.


create Session object
```python
sess = tf.Session()
```

Initialise the variables
```python
sess.run(tf.global_variables_initializer())
```
We need to initialise the variables as we only assign real values to the variables once we are in runtime (i.e. after the session is created.)

**The run method**
The run method of Session object evaluates the output of a node of the graph. It takes two arguments:
```python
sess.run(fetches, feeds)
```
`fetches` is a list of graph nodes that are evaluated. The Session returns the output of these nodes.

`feeds` are the Dictionary mappings from graph nodes to concrete values. Feed Dictionaries specify the value of each Placeholder required by the node to manipulate the data.
In order words, ...
* The name of the keys of Feed Dictionaryies should be the same as the *Placeholders*.


### Visualisation using TensorBoard
* To allow for visualisation, we first have to create a FileWriter object for visualising the training logs, which need to be stored before utilising TensorBoard.
```python
writer = tf.summary.FileWriter('./graphs', graph=tf.get_default_graph())
```
Storing our logs to `./graphs` directory.

**Logging dynamic values**
* the fluctuating loss values and accurary.
* parameters

for scalars:
```python
tf.summary.scalar("loss", loss)
tf.summary.scalar("accuracy", accuracy)
```

For parameters, we can create histogram (can't we use something other than histogram?) summaries:
```python
tf.summary.histogram("weights", weights)
tf.summary.histogram("biases", biases)
```
These histograms diaplyar the occurence of number of values relative to other value.
They are *very helpful* in studying the distribution of a parameter values over time.

**Merging Summary Operations**
We can merge these summary operations so that they can be *executed as a single operation* inside the session
```python
summary_op = tf.summary.merge_all()`
```

**Running and logging the summary operation**
After merging the summaries, we can run the Summary operation inside the session, and write its output to our FileWriter.
Note that `i` is the iteration index inside the training loop (training step)
```python
_, summary = sess.run([train_step, summary_op], feed_dict)
writer.add_summary(summary, i)
```

### Making the graph readable
By default the graph looks very messy

You are able to label the graph by adding a name scope to nodes
```python
with tf.name_scope('LinearModel'):
    logits = tf.matmul(x, weights) + biases
    y_pred = tf.nn.softmax(logits)
```
This will annotate the graph nodes (box them and put a label) and makes the graph readable.
* you can always click on the `+` to view the full underlying graph.
* It's essentially a code folding mechanism but for graphs

We can also name our placeholder and variable nodes:
```python
x = tf.placeholder(tf.float32, [None, 784, name='x'))
y_true = tf.placeholder(tf.float32, [None,10], name='labels')

weights= tf.Variable(tf.random_uniform([784,10],-1,1), name='weights')
biases = tf.Variable(tf.zeros([10], name='biases')
```
![alt text](https://user-images.githubusercontent.com/11167307/34468284-ba2e1dba-eefd-11e7-8473-42c2e83397c9.png)

### Resources for Tensorflow
[https://www.slideshare.net/tw_dsconf/tensorflow-tutorial](https://www.slideshare.net/tw_dsconf/tensorflow-tutorial)













### How to activate tensorboard
I will most likely have to activate tensorboard on my home pc. To see the build of my computational graph.
Since this doesn't require any execution I should be okay.

[source](https://deepnotes.io/tensorflow)
