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
    Returns: A tensor of shape (N_examples, 43), with values equal to the logits of
      classifying the object images into one of 43 classes
      (specific roadsign, another roadsign, etc)
    """

    #Initializations define the way to set the initial random weights to the layers
    initializer = tf.random_uniform_initializer(minval=-0.05,
    maxval=0.05)
    #initializer = tf.contrib.layers.xavier_initializer()

    # First convolutional layer - maps one RGB image to 32 feature maps.
    conv1 = tf.layers.conv2d(
        inputs=x_image,
        filters=32,
        kernel_size=[5, 5],
	    kernel_initializer=initializer,
	    strides=1,
        padding='same',
        name='conv1'
    )
    conv1_bn = tf.layers.batch_normalization(conv1, name='conv1_bn')
    pool1 = tf.layers.average_pooling2d(
        inputs=conv1_bn,
        pool_size=[3, 3],
        strides=2,
        padding='same',
        name='pool1'
    )

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=32,
        kernel_size=[5, 5],
	    kernel_initializer=initializer,
        strides=1,
        padding='same',
        name='conv2'
    )
    conv2_bn = tf.layers.batch_normalization(conv2, name='conv2_bn')
    pool2 = tf.layers.average_pooling2d(
        inputs=conv2_bn,
        pool_size=[3, 3],
        strides=2,
        padding='same',
        name='pool2'
    )

    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=64,
        kernel_size=[5, 5],
	    kernel_initializer=initializer,
        strides=1,
        padding='same',
        name='conv3'
    )
    conv3_bn = tf.layers.batch_normalization(conv3, name='conv3_bn')
    pool3 = tf.layers.max_pooling2d(
        inputs=conv3_bn,
        pool_size=[3, 3],
        strides=2,
        padding='same',
        name='pool3'
    )

    pool3_flat = tf.reshape(pool3, [-1, 4 * 4 * 64], name='pool3_flattened')

    fc1 = tf.layers.dense(inputs=pool3_flat, activation=tf.nn.relu, units=64, name='fc1')
    fc2 = tf.layers.dense(inputs=fc1, activation=tf.nn.relu, units=64, name='fc2')

    logits = tf.layers.dense(inputs=fc2, units=class_count, name='logit')
    return logits

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

def main(_):
    tf.reset_default_graph()

    # load data
    data = pickle.load(open('dataset.pkl', 'rb'))
    
    # load in whole data set to calucalte statistical values
    train = data[0]
    train_images = [train[i][0] for i in range(0,39209)]
    
    # Calculate rgb tuple of mean and std
    train_mean, train_std = calculate_mean_std(train_images)
    
    with tf.variable_scope('inputs'):
        # Create the model
        x = tf.placeholder(tf.float32, [None, FLAGS.img_width * FLAGS.img_height * FLAGS.img_channels])
        x_image = tf.reshape(x, [-1, FLAGS.img_width, FLAGS.img_height, FLAGS.img_channels])
        y_ = tf.placeholder(tf.float32, [None, FLAGS.num_classes])

	# subtract the mean and standard deviation
	#x_image = tf.map_fn(lambda x: whitening(x, train_mean, train_std), x_image)

    with tf.variable_scope('model'):
        logits = deepnn(x_image)


        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
        
        # add all the variables you want to add weight decay to, to a collection name 'variables' 
        # and then you calculate the L2 norm weight decay for the whole collection. 
        # create variable as a collection of weights
        
        
        # weights =  tf.get_variable('weights', collections=['weights'])
        weight_decay = tf.constant(0.0001, dtype=tf.float32) # your weight decay rate, must be a scalar tensor.
        
        # weights_norm = tf.reduce_sum(
        #   input_tensor= weight_decay * tf.stack(
        #         [tf.nn.l2_loss(i) for i in tf.get_collection('weights')]
        #     ),
        #     name='weights_norm'
        # )
            
            
        # the weights are not working...
        # weights1 = tf.get_collection(tf.GraphKeys.VARIABLES, 'conv1/kernel')
        # weights2 = tf.get_collection(tf.GraphKeys.VARIABLES, 'conv2/kernel')
        # weights3 = tf.get_collection(tf.GraphKeys.VARIABLES, 'conv3/kernel')
       
       
        #W = tf.get_variable(name='weight', shape=[4, 4, 256, 512], regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
        #https://stackoverflow.com/questions/36570904/how-to-define-weight-decay-for-individual-layers-in-tensorflow/36573850#36573850
        
        # with L2 test 2 (think this works)
        # trainable_vars   = tf.trainable_variables() # should be weight alone
        # lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in trainable_vars if 'bias' not in v.name ]) * weight_decay
        # cost = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)) + lossL2)
        
        
        
        global_step = tf.Variable(0, trainable=False)

        train_step = tf.train.MomentumOptimizer(learning_rate=FLAGS.learning_rate,
                                                momentum=0.9).minimize(loss=cross_entropy, global_step=global_step)

    # summaries
    loss_summary = tf.summary.scalar("Loss", cross_entropy)
    accuracy_summary = tf.summary.scalar("Accuracy", accuracy)
    # learning_rate_summary = tf.summary.scalar("Learning Rate", FLAGS.learning_rate)
    img_summary = tf.summary.image('Input Images', x_image)
    test_img_summary = tf.summary.image('Test Images', x_image)
    
    train_summary = tf.summary.merge([loss_summary, accuracy_summary, img_summary])
    validation_summary = tf.summary.merge([loss_summary, accuracy_summary])
    # test_summary = tf.summary.merge([img_summary, accuracy_summary])
    test_summary = tf.summary.merge([test_img_summary])

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
       
       
        # define summaries for tf.summary.FileWriter 
        # adv_x_summary = tf.summary.image('Adversarial Images', adv_x) image not defined don't use for now

        train_writer = tf.summary.FileWriter(run_log_dir + "_train", sess.graph)
        validation_writer = tf.summary.FileWriter(run_log_dir + "_validation", sess.graph)
        white_image_train_writer = tf.summary.FileWriter(run_log_dir + "_train_white", sess.graph)
        test_writer = tf.summary.FileWriter(run_log_dir + "_test", sess.graph)
        test_whiten_writer = tf.summary.FileWriter(run_log_dir + "_test_whiten", sess.graph)

        
        # Train and validation

        #Training for one epoch using 392 batches of size 100
        step = 0 #if this run we don't need it
        no_validation_decreases = 0

        # validation images and its labels
        data_validation = data[1]
        validation_images = [data_validation[i][0] for i in range(0, 12630)]
        validation_labels = [data_validation[i][1] for i in range(0,12630)]
        
        # lets not whiten the validation for now
        # whiten validation images
        # validation_images = whitening(validation_images, train_mean, train_std)
        prev_validation_accuracy = 0

        for _ in range(FLAGS.training_epochs):
            train = batch_generator(data,'train')
            for train_images, train_labels in train:
                
                train_images_whiten = whitening(train_images, train_mean, train_std)
                
                if DEBUG:
                    # check if whitening is working
                    print(train_images[0][0][0])
                    print('-----')
                    print(train_images_whiten[0][0][0])
                    print('another image')
                
                _, train_summary_str = sess.run([train_step, train_summary],
                                                feed_dict={x_image: train_images, y_: train_labels})
                
                _, train_summary_whiten_str = sess.run([train_step, train_summary],
                                                feed_dict={x_image: train_images_whiten, y_: train_labels})
                                                
                # havne't implemented weight decay yet I need to see the weight variable value first before implementing decay
                # a = sess.run([weights1, weights2, weights3])
                # print(a)   
                
                # Validation: Monitoring accuracy using validation set
                
                # Only add summary when step and log frequency are divisible
                if step % FLAGS.log_frequency == 0:
                    #validation_accuracy = 0
                    #batch_count = 0
                    #validation = batch_generator(data, 'test')
                    #for (test_images, test_labels) in validation:
                    

                    train_writer.add_summary(train_summary_str, step) 
                    white_image_train_writer.add_summary(train_summary_whiten_str, step)
                    
                    validation_accuracy, validation_summary_str = sess.run([accuracy, validation_summary],
                                                                            feed_dict={x_image: validation_images, y_: validation_labels})
                    
                    #batch_count += 1
                    #validation_accuracy += validation_accuracy_temp
                    #validation_accuracy = validation_accuracy / batch_count
                        
                    if (validation_accuracy - prev_validation_accuracy) <= 0 and num_learning_rate_decreases < 3:
                        # check accuracy is no longer improving / getting worse
                        # decrease the learning rate 
                        # don't decrease the learning rate anymore than 3 times
                        num_learning_rate_decreases +=1
                        FLAGS.learning_rate = FLAGS.learning_rate/10
                  
                    prev_validation_accuracy = validation_accuracy

                     
                    
                    print('step {}, accuracy on validation set : {}'.format(step, validation_accuracy))
                    validation_writer.add_summary(validation_summary_str, step)
                    

                # Save the model checkpoint periodically.
                if step % FLAGS.save_model_frequency == 0 or (step + 1) == FLAGS.max_steps:
                # if (step + 1) % FLAGS.save_model_frequency == 0 or (step + 1) == FLAGS.max_steps:
                    saver.save(sess, checkpoint_path, global_step=step)

                if step % FLAGS.flush_frequency == 0:
                # if (step + 1) % FLAGS.flush_frequency == 0:
                    train_writer.flush()
                    validation_writer.flush()
                    white_image_train_writer.flush()
                    
                step += 1

            
               
        # Testing
        
        # read in the entire test set rather than batch
        data_test = data[1]
        test_images = [data_test[i][0] for i in range(0, 12630)]
        test_labels = [data_test[i][1] for i in range(0,12630)]
        test_whiten_images = whitening(test_images, train_mean, train_std)
        
        test_images = sess.run(test_images, feed_dict={x: test_images})
        test_whiten_images = sess.run(test_white_images, feed_dict={x_image: test_whiten_images})
        
        test_accuracy, test_summary_str = sess.run([accuracy, test_summary],
                                                    feed_dict={x_image: test_images, y_: test_labels})
                                                    
        test_whiten_accuracy, test_whiten_summary_str = sess.run([accuracy, test_summary], 
                                                                feed_dict={x_image: test_whiten_images, y_: test_labels})

        #test = batch_generator(data, 'test')
        #for (test_images, test_labels) in test:
            #test_accuracy_temp, _ = sess.run([accuracy, test_summary], feed_dict={x_image: test_images, y_: test_labels})
        
            #batch_count += 1
            #test_accuracy += test_accuracy_temp

        #test_accuracy = test_accuracy / batch_count
        
        test_writer.add_summary(test_summary_str, step)
        test_whiten_writer.add_summary(test_whiten_summary_str, step)
        
        test_writer.flush()
        test_whiten_writer.flush()
        
        print('test set: accuracy on test set: %0.3f' % test_accuracy)
        print('test whitened set: accuracy on test set: %0.3f' % test_whiten_accuracy)

        print('model saved to ' + checkpoint_path)

        # close FileWriters
        train_writer.close()
        validation_writer.close()
        white_image_train_writer.close()
        test_writer.close()
        test_whiten_writer.close()

if __name__ == '__main__':
    tf.app.run(main=main)
