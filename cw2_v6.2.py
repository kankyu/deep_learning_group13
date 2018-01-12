#Seventh Version

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import os
import os.path

import tensorflow as tf
import numpy as np
from gtsrb import batch_generator
import cPickle as pickle

FLAGS = tf.app.flags.FLAGS
DEBUG = False

# generate graphs with training images whitened
# training images are whitened
# validation images are whitened, note we overwrite the original validation images.
# due to implementation restrictions, when you run the program you have the choice of whiten the validation image or not.
# test images are whitened 
whiten = True
# batch_normalise = True not implemented

tf.app.flags.DEFINE_integer('training-epochs', 45,
                            'Maximum number of training epochs to train. (default: %(default)d)')
tf.app.flags.DEFINE_integer('batch-size', 100, 'Number of examples per mini-batch (default: %(default)d)')
# all figures use 45 epochs and tilo said in forum to stop after 45 epochs
#45*393=17685
tf.app.flags.DEFINE_integer('max-steps', 17685,
                            'Number of mini-batches to train on. (default: %(default)d)')
tf.app.flags.DEFINE_integer('log-frequency', 50,
                            'Number of steps between logging results to the console and saving summaries (default: %(default)d)')
tf.app.flags.DEFINE_integer('save-model-frequency', 100,
                            'Number of steps between model saves (default: %(default)d)')
tf.app.flags.DEFINE_integer('flush-frequency', 50,
                            'Number of steps between flushing summary results. (default: %(default)d)')
tf.app.flags.DEFINE_float('learning-rate', 0.01, 'Number of examples to run. (default: %(default)d)')
tf.app.flags.DEFINE_string('log-dir', '{cwd}/logs/'.format(cwd=os.getcwd()),
                           'Directory where to write event logs and checkpoint. (default: %(default)s)')
tf.app.flags.DEFINE_integer('img-width', 32, 'Image width (default: %(default)d)')
tf.app.flags.DEFINE_integer('img-height', 32, 'Image height (default: %(default)d)')
tf.app.flags.DEFINE_integer('img-channels', 3, 'Image channels (default: %(default)d)')
tf.app.flags.DEFINE_integer('num-classes', 43, 'Number of classes (default: %(default)d)')


run_log_dir = os.path.join(FLAGS.log_dir, 'exp_bs_{bs}_lr_{lr}'.format(bs=FLAGS.batch_size,
                                                                       lr=FLAGS.learning_rate))
checkpoint_path = os.path.join(run_log_dir, 'model.ckpt')


def deepnn(x_image, class_count=43):
    """deepnn builds the graph for a deep net for classifying images.
    Args:
        x_image: an input tensor whose ``shape[1:] = img_space``
            (i.e. a batch of images conforming to the shape specified in ``img_shape``)
        class_count: number of classes in dataset
    Returns:
        y: is a tensor of shape (N_examples, 43), with values
          equal to the logits of classifying the object images into one of 43 classes
      (specific roadsign, another roadsign, etc)
        img_summary: a string tensor containing sampled input images.
    """

    img_summary = tf.summary.image('Input_images', x_image)

    # Small epsilon value for the BN transform
    epsilon = 1e-3

    # First convolutional layer - maps one image to 32 feature maps.
    with tf.variable_scope('Conv_1'):
        W_conv1 = weight_variable([5, 5, FLAGS.img_channels, 32])
        #Note bias is not needed when using batch normalisation
        #Also apply relu function after batch normalisation
        b_conv1 = bias_variable([32])
        conv1_bn = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        
        # Turn of batch normalisation. Comment out below up to h_pool1 not including hpool1

        h_conv1 = conv2d(x_image, W_conv1)
        batch_mean1, batch_var1 = tf.nn.moments(h_conv1, [0])
        gamma1 = tf.Variable(tf.ones([32]))
        beta1 = tf.Variable(tf.zeros([32]))
        conv1_bn = tf.nn.relu(tf.nn.batch_normalization(h_conv1, batch_mean1, batch_var1, beta1, gamma1, epsilon))

        # Pooling layer - downsamples by 2X.
        h_pool1 = avg_pool_2x2(conv1_bn)

    with tf.variable_scope('Conv_2'):
        # Second convolutional layer -- maps 32 feature maps to 32.
        W_conv2 = weight_variable([5, 5, 32, 32])
        b_conv2 = bias_variable([32])
        
        conv2_bn = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

        # Turn off Batch normalisation. Comment out below up to h_pool1 not including hpool2
        
        h_conv2 = conv2d(h_pool1, W_conv2)
        batch_mean2, batch_var2 = tf.nn.moments(h_conv2, [0])
        gamma2 = tf.Variable(tf.ones([32]))
        beta2 = tf.Variable(tf.zeros([32]))
        conv2_bn = tf.nn.relu(tf.nn.batch_normalization(h_conv2, batch_mean2, batch_var2, beta2, gamma2, epsilon))

        # Second pooling layer.
        h_pool2 = avg_pool_2x2(conv2_bn)

    with tf.variable_scope('Conv_3'):
        # Third convolutional layer -- maps 32 feature maps to 64.
        W_conv3 = weight_variable([5, 5, 32, 64])
        b_conv3 = bias_variable([64])
        
        conv3_bn = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

        # Turn off Batch normalisation. Comment out below up to h_pool1 not including hpool3

        h_conv3 = conv2d(h_pool2, W_conv3)
        batch_mean3, batch_var3 = tf.nn.moments(h_conv3, [0])
        gamma3 = tf.Variable(tf.ones([64]))
        beta3 = tf.Variable(tf.zeros([64]))
        conv3_bn = tf.nn.relu(tf.nn.batch_normalization(h_conv3, batch_mean3, batch_var3, beta3, gamma3, epsilon))

        # Third pooling layer.
        h_pool3 = max_pool_2x2(conv3_bn)
    with tf.variable_scope('FC_1'):
        # Fully connected layer 1 -- after 2 round of downsampling, our 32x32
        # image is down to 4x4x64 feature maps -- maps this to 1024 features.
        W_fc1 = weight_variable([4 * 4 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool3_flat = tf.reshape(h_pool3, [-1, 4*4*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1)+ b_fc1)

    with tf.variable_scope('FC_2'):
        # Second fully-connected layer
        W_fc2 = weight_variable([1024, 1024])
        b_fc2 = bias_variable([1024])

        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    with tf.variable_scope('FC_3'):
        # Map the 1024 features to 43 classes
        W_fc3 = weight_variable([1024, FLAGS.num_classes])
        b_fc3 = bias_variable([FLAGS.num_classes])

        y_conv = tf.matmul(h_fc2, W_fc3)
        return y_conv, img_summary


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name='convolution')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                          strides=[1, 2, 2, 1], padding='SAME', name='max_pooling')

def avg_pool_2x2(x):
    """avg_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.avg_pool(x, ksize=[1, 3, 3, 1],
                          strides=[1, 2, 2, 1], padding='SAME', name='avg_pooling')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.random_uniform(
    shape,
    minval=-0.05,
    maxval=0.05)
    #initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name='weights')
    #Weight = tf.get_variable("weights", shape=shape,
			#initializer=tf.contrib.layers.xavier_initializer())
    #return Weight


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='biases')
    #Bias = tf.get_variable("biases", shape=shape,
	#		initializer=tf.contrib.layers.xavier_initializer())
    #return Bias


def whitening(image, mean, std):
    """ 
    subtract the mean divide by standard deviation
    img_whitten = (image-mean)/std
    """
    a = np.array(image)
    img_whiten = np.divide(np.subtract(a, mean), std)
    return img_whiten

def calculate_mean_std(images):
    a = np.array(images)
    b = a.sum(axis=0)
    c = b.sum(axis=0)
    d = c.sum(axis=0)
    N = 32 * 32 * 39209
    mean = d/N

    squares = np.square(a)
    squares_b = squares.sum(axis=0)
    squares_c = squares_b.sum(axis=0)
    squares_d = squares_c.sum(axis=0)
    variance = squares_d/N - np.square(mean)
    std = np.sqrt(variance)

    return mean, std

def decrease(n_steps=5):
    """ 
    n_steps number of consecutive steps accuracy has decreased.
    condition in which we want the learning rate to decay
    returns: boolean
    """
    
    """ old method
    if (validation_accuracy - prev_validation_accuracy) <= 0:  and num_learning_rate_decreases < 3:
        # check accuracy is no longer improving / getting worse
        # decrease the learning rate 
        # don't decrease the learning rate anymore than 3 times
        num_learning_rate_decreases +=1
        FLAGS.learning_rate = FLAGS.learning_rate/10
                  
    prev_validation_accuracy = validation_accuracy
    """
    pass

def main(_):
    tf.reset_default_graph()

    # load data
    data = pickle.load(open('dataset.pkl', 'rb'))
    
    # load in whole data set to calucalte statistical values
    train = data[0]
    train_images = [train[i][0] for i in range(0,39209)]

    # Calculate rgb tuple of mean and std
    if whiten:
        train_mean, train_std = calculate_mean_std(train_images)

    with tf.variable_scope('inputs'):
        # Create the model
        x = tf.placeholder(tf.float32, [None, FLAGS.img_width * FLAGS.img_height * FLAGS.img_channels])
        x_image = tf.reshape(x, [-1, FLAGS.img_width, FLAGS.img_height, FLAGS.img_channels])
        y_ = tf.placeholder(tf.float32, [None, FLAGS.num_classes])


    with tf.variable_scope('model'):
        # Build the graph for the deep net
        y_conv, img_summary = deepnn(x_image)


        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
        global_step = tf.Variable(0, trainable=False)

        train_step = tf.train.MomentumOptimizer(learning_rate=FLAGS.learning_rate,
                                                momentum=0.9).minimize(loss=cross_entropy, global_step=global_step)

    # summaries
    #learning_rate_summary = tf.summary.scalar("Learning Rate", decayed_learning_rate)
    #img_summary = tf.summary.image('Input Images', x_image)

    accuracy_summary = tf.summary.scalar("Accuracy", accuracy)
    loss_summary = tf.summary.scalar("Loss", cross_entropy)
    error_summary = tf.summary.scalar("Error", 1-accuracy)
    # learning_rate_summary = tf.summary.scalar("Learning Rate", FLAGS.learning_rate)
    img_summary = tf.summary.image('Input Images', x_image)
    test_img_summary = tf.summary.image('Test Images', x_image)
    
    train_summary = tf.summary.merge([loss_summary, accuracy_summary, error_summary, img_summary])
    validation_summary = tf.summary.merge([loss_summary, accuracy_summary, error_summary])
    # test_summary = tf.summary.merge([img_summary, accuracy_summary])
    test_summary = tf.summary.merge([test_img_summary])

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
       
       
        # define summaries for tf.summary.FileWriter 
        # adv_x_summary = tf.summary.image('Adversarial Images', adv_x) image not defined don't use for now

        train_writer = tf.summary.FileWriter(run_log_dir + "_train", sess.graph)
        validation_writer = tf.summary.FileWriter(run_log_dir + "_validation", sess.graph)
        test_writer = tf.summary.FileWriter(run_log_dir + "_test", sess.graph)
        if whiten:
            white_image_train_writer = tf.summary.FileWriter(run_log_dir + "_train_white", sess.graph)
            test_whiten_writer = tf.summary.FileWriter(run_log_dir + "_test_whiten", sess.graph)
        
        # Train and validation

        #Training for one epoch using 392 batches of size 100
        step = 0 #if this run we don't need it
        no_valiation_decreases = 0

        # validation images and its labels
        data_validation = data[1]
        validation_images = [data_validation[i][0] for i in range(0, 12630)]
        validation_labels = [data_validation[i][1] for i in range(0,12630)]
        
        
        # comment if you don't want to whiten validation images, it overwrites the normal validation images
        validation_images = whitening(validation_images, train_mean, train_std)
            
        num_learning_rate_decreases = 0
        
        # Number of values to avg for decay rate. I think this can be better explained.
        num_avg = 4 
        prev_validation_accuracy = np.zeros(num_avg)
        validation_acc = np.zeros(num_avg)
                
        for _ in range(FLAGS.training_epochs):
            train = batch_generator(data,'train')
            for train_images, train_labels in train:
                
                epoch_step = step/393
                
                if whiten:
                    train_images_whiten = whitening(train_images, train_mean, train_std)
                
                if DEBUG and whiten:
                    # check if whitening is working
                    print(train_images[0][0][0])
                    print('-----')
                    print(train_images_whiten[0][0][0])
                    print('another image')
                
                _, train_summary_str = sess.run([train_step, train_summary],
                                                feed_dict={x_image: train_images, y_: train_labels})
                
                if whiten:
                    _, train_summary_whiten_str = sess.run([train_step, train_summary],
                                                            feed_dict={x_image: train_images_whiten, y_: train_labels})
                                                            
                # havne't implemented weight decay yet I need to see the weight variable value first before implementing decay
                
                # Validation: Monitoring accuracy using validation set
                
                # Only add summary when step and log frequency are divisible
                if step % FLAGS.log_frequency == 0:
                    #validation_accuracy = 0
                    #batch_count = 0
                    #validation = batch_generator(data, 'test')
                    #for (test_images, test_labels) in validation:

                    train_writer.add_summary(train_summary_str, epoch_step) 
                    
                    if whiten: 
                        white_image_train_writer.add_summary(train_summary_whiten_str, epoch_step)
                    
                    validation_accuracy, validation_summary_str = sess.run([accuracy, validation_summary],
                                                                            feed_dict={x_image: validation_images, y_: validation_labels})
                    
                    validation_acc[step % num_avg] = validation_accuracy
                    
                    #batch_count += 1
                    #validation_accuracy += validation_accuracy_temp
                    #validation_accuracy = validation_accuracy / batch_count
                        
                    if np.mean(validation_acc - prev_validation_accuracy) <= 0 and num_learning_rate_decreases < 3:
                        # check if it has been decreasing on average of 5 steps
                        num_learning_rate_decreases += 1
                        FLAGS.learning_rate = FLAGS.learning_rate/10
                  
                    prev_validation_accuracy[step % num_avg] = validation_accuracy

                    print('step {}, accuracy on validation set : {}'.format(step, validation_accuracy))
                    validation_writer.add_summary(validation_summary_str, epoch_step)

                # Save the model checkpoint periodically.
                if step % FLAGS.save_model_frequency == 0 or (step + 1) == FLAGS.max_steps:
                # if (step + 1) % FLAGS.save_model_frequency == 0 or (step + 1) == FLAGS.max_steps:
                    saver.save(sess, checkpoint_path, global_step=step)

                if step % FLAGS.flush_frequency == 0:
                # if (step + 1) % FLAGS.flush_frequency == 0:
                    train_writer.flush()
                    validation_writer.flush()
                    if whiten:
                        white_image_train_writer.flush()
                    
                step += 1
               
        # Testing
        
        epoch_step = step/393
        # read in the entire test set rather than batch
        data_test = data[1]
        test_images = [data_test[i][0] for i in range(0, 12630)]
        test_labels = [data_test[i][1] for i in range(0,12630)]
        
        # test_images = sess.run(test_images, feed_dict={x_image: test_images})
        test_accuracy, test_summary_str = sess.run([accuracy, test_summary],
                                                    feed_dict={x_image: test_images, y_: test_labels})
                                                    
        test_writer.add_summary(test_summary_str, epoch_step)
        test_writer.flush()
        print('test set: accuracy on test set: %0.3f' % test_accuracy)
                                                    
        if whiten: 
            test_whiten_images = whitening(test_images, train_mean, train_std)
            test_whiten_images = sess.run(test_white_images, feed_dict={x_image: test_whiten_images})
            test_whiten_accuracy, test_whiten_summary_str = sess.run([accuracy, test_summary], 
                                                                feed_dict={x_image: test_whiten_images, y_: test_labels})
            test_whiten_writer.add_summary(test_whiten_summary_str, epoch_step) # step/393 to get the epoch step
            test_whiten_writer.flush()
            print('test whitened set: accuracy on test set: %0.3f' % test_whiten_accuracy)

        print('model saved to ' + checkpoint_path)

        # close FileWriters
        train_writer.close()
        validation_writer.close()
        test_writer.close()
        
        if whiten:
            white_image_train_writer.close()
            test_whiten_writer.close()

if __name__ == '__main__':
    tf.app.run(main=main)
