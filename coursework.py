from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

import numpy as np
import tensorflow as tf

here = os.path.dirname(__file__)

dataset_file = 'gtsrb_dataset.npz'
data = np.load(dataset_file)

# settings
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('log-frequency', 10,
                            'Number of steps between logging results to the console and saving summaries.' +
                            ' (default: %(default)d)')
tf.app.flags.DEFINE_integer('flush-frequency', 50,
                            'Number of steps between flushing summary results. (default: %(default)d)')
tf.app.flags.DEFINE_integer('save-model-frequency', 100,
                            'Number of steps between model saves. (default: %(default)d)')
tf.app.flags.DEFINE_string('log-dir', '{cwd}/logs/'.format(cwd=here),
                           'Directory where to write event logs and checkpoint. (default: %(default)s)')

# Optimisation hyperparameters
# for coding purposes set max_steps to a small value
max_steps_real = 10000
max_steps = 1000
tf.app.flags.DEFINE_integer('max-steps', max_steps,
                            'Number of mini-batches to train on. (default: %(default)d)')
tf.app.flags.DEFINE_integer('batch-size', 100, 'Number of examples per mini-batch. (default: %(default)d)')
tf.app.flags.DEFINE_float('learning-rate', 1e-3, 'Number of examples to run. (default: %(default)d)')
tf.app.flags.DEFINE_integer('img-width', 32, 'Image width (default: %(default)d)')
tf.app.flags.DEFINE_integer('img-height', 32, 'Image height (default: %(default)d)')
tf.app.flags.DEFINE_integer('img-channels', 3, 'Image channels (default: %(default)d)')
tf.app.flags.DEFINE_integer('num-classes', 43, 'Number of classes (default: %(default)d)')

sess = tf.InteractiveSession()


def main():
    # clear graph
    tf.reset_default_graph()

    for x_train, y_train in batch_generator(data, 'train'):
        pass

    # Build the graph
    with tf.name_scope('inputs'):
        x = tf.placeholder(tf.float32, shape=[None, Flags.image_width * Flags.image_height * Flags.image_channel])
        
        # for batch_size is dynamically computed based input values
        x_image = tf.reshape(x, [-1, 32, 32, 3])

        # what is class count, i think its the len of the vector
        y_ = tf.placeholder(tf.float32, shape=[None, Flag.num_classes]

    with tf.variable_scope('model'):
        logits = deep_nn(x_image, Flag.num_classes)
        
        # https://deepnotes.io/softmax-crossentropy
        # cross_entropy 
        cross_entropy_tensor = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits)

        #assert type(cross_entropy_tensor) == #logit object but can't find definition
        
        # compute the mean of elements across (all) dimensions of a tensor
        cross_entropy_loss = tf.reduce_mean(cross_entropy_tensor)

        decay_steps = 1000 # decay every 1000 steps
        decay_rate = 0.8 # the base of our exponential for the decay
        global_step = tf.Variable(0, trainable=False) # incremented by Tensorflow
        # reduce the learning rate every step
        decayed_learning_rate = tf.train.exponential_decay(Flags.learning_rate, global_step,
                                                            decay_steps, decay_rate, staircase=True)

        


        

def deep_nn(x_image, class_count):
    """ model for our CNN """
    # https://www.tensorflow.org/tutorials/layers 


    # first convolutional layer - maps on RBG image to 32 feature maps
    conv1 = tf.layers.conv2d(
        inputs = x_image,
        filters = 32,
        kernel_size = [5,5],
        padding = 'same',
        use_bias = False,
        name='conv1'
        )

    # normalise batch
    conv1_bn = tf.layers.batch_normalization(conv1, name=conv1_bn)
    # apply activation 
    conv1_bn = tf.nn.relu(conv1_bn)

    # pool layer 1
    pool1 = tf.layers.max_pooling2d(
        inputs = conv1_bn,
        pool_size = [2,2],
        strides = 2,
        name = 'pool1'
        )

    # convolutational layer 2
    conv2 = tf.layers.conv2d(
            inputs = pool1,
            filters = 64,
            kernel_size = [5,5],
            padding = 'same',
            activation = tf.nn.relu,
            use_bias = False,
            name = 'conv2'
            )

    # normalise batch
    conv2_bn = tf.layers.batch_normalization(conv2, name=conv2_bn)
    # apply activation 
    conv2_bn = tf.nn.relu(conv2_bn)

    # pool layer 2
    pool2 = tf.layers.max_pooling2d(
            inputs = conv2_bn,
            pool_size = [2,2],
            strides = 2,
            name = 'pool2'
            )

    pool2.eval()
    exit()
    
    # dense layer, i'm not how to determine the size.
    #pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64], name='pool2_flattenned')
    
    # fully connected layer 1
    # unit? look up
    fc1 = tf.layers.dense(
            inputs = pool2_flat,
            activation = tf.nn.relu,
            units = 1024,
            name = 'fc1'
            )
    
    # fully connected layer 2 and assigned as logits

    logits = tf.layers.dense(
            inputs = fc1,
            units = class_count,
            name = 'fc2'
            )

    return logits
    
    # put batch of images into tf
    # make the neural network

        
if __name__ == '__main__':
    tf.app.run()