#Sixth Attempt

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

# tilo said to use 45 epochs because that is what all the figures have
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

def whitening(image):
    """ subtract the mean and standard deviation """
    return tf.image.per_image_standardization(image)
    
def main(_):
    tf.reset_default_graph()

    #Import Data
    data = np.load('gtsrb_dataset.npz')
    #data = pickle.load(open('dataset.pkl', 'rb'))

    with tf.variable_scope('inputs'):
        # Create the model
        x = tf.placeholder(tf.float32, [None, FLAGS.img_width * FLAGS.img_height * FLAGS.img_channels])
        x_image = tf.reshape(x, [-1, FLAGS.img_width, FLAGS.img_height, FLAGS.img_channels])
        y_ = tf.placeholder(tf.float32, [None, FLAGS.num_classes])
        
        # subtract the mean and deviation 
        x_image = tf.map_fn(lambda x: whitening(x), x_image)
        
        # maybe we will need this.
        # weight = tf.Variable(<initial-value>, name='Weight')

    with tf.variable_scope('model'):
        logits = deepnn(x_image)


        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
        
        #weights1 = tf.get_collection(tf.GraphKeys.VARIABLES, 'conv1/kernel')[0]
        #weights2 = tf.get_collection(tf.GraphKeys.VARIABLES, 'conv2/kernel')[0]
        #weights3 = tf.get_collection(tf.GraphKeys.VARIABLES, 'conv3/kernel')[0]
        
        #weight_decay = tf.constant(0.0001, dtype=tf.float32) # your weight decay rate, must be a scalar tensor.
        #W = tf.get_variable(name='weight', shape=[4, 4, 256, 512], regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
        #https://stackoverflow.com/questions/36570904/how-to-define-weight-decay-for-individual-layers-in-tensorflow/36573850#36573850
        
        train_step = tf.train.MomentumOptimizer(learning_rate=FLAG.learning_rate, momentum=0.9).minimize(loss=cross_entropy, global_step=global_step)

    loss_summary = tf.summary.scalar("Loss", cross_entropy)
    accuracy_summary = tf.summary.scalar("Accuracy", accuracy)
    #learning_rate_summary = tf.summary.scalar("Learning Rate", decayed_learning_rate)
    img_summary = tf.summary.image('Input Images', x_image)

    # Summaries for TensorBoard visualisation
    train_summary = tf.summary.merge([loss_summary, accuracy_summary, img_summary])
    validation_summary = tf.summary.merge([loss_summary, accuracy_summary])
    test_summary = tf.summary.merge([img_summary, accuracy_summary])

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(run_log_dir + "_train", sess.graph)
        validation_writer = tf.summary.FileWriter(run_log_dir + "_validation", sess.graph)

        sess.run(tf.global_variables_initializer())

        #Training for one epoch using 392 batches of size 100
        step = 0

        # Setup the validation images and labels
        data_validation = data[1]
        validation_images = [data_validation[i][0] for i in range(0, 12630)]
        validation_labels = [data_validation[i][1] for i in range(0, 12630)]

        for i in range(FLAGS.training_epochs):
            train = batch_generator(data,'train')
            for (train_images, train_labels) in train:
                _, train_summary_str = sess.run([train_step, train_summary],
                                                feed_dict={x_image: train_images, y_: train_labels})

                step += 1

                # Validation: Monitoring accuracy using validation set
                if step % FLAGS.log_frequency == 0:
                    #validation_accuracy = 0
                    #batch_count = 0
                    #validation = batch_generator(data, 'test')
                    #for (test_images, test_labels) in validation:

                    validation_accuracy, validation_summary_str = sess.run([accuracy, validation_summary],
                                                                            feed_dict={x_image: validation_images, y_: validation_labels})

                        #batch_count += 1
                        #validation_accuracy += validation_accuracy_temp

                    #validation_accuracy = validation_accuracy / batch_count

                    print('step {}, accuracy on validation set : {}'.format(step, validation_accuracy))
                    validation_writer.add_summary(validation_summary_str, step)

                # Save the model checkpoint periodically.
                if (step + 1) % FLAGS.save_model_frequency == 0 or (step + 1) == FLAGS.max_steps:
                    saver.save(sess, checkpoint_path, global_step=step)

                if (step + 1) % FLAGS.flush_frequency == 0:
                    train_writer.flush()
                    validation_writer.flush()


        test_accuracy = 0
        #batch_count = 0

        data_test = data[1]
        test_images = [data_test[i][0] for i in range(0, 12630)]
        test_labels = [data_test[i][1] for i in range(0, 12630)]
        test_accuracy, _ = sess.run([accuracy, test_summary], feed_dict={x_image: test_images, y_: test_labels})


        #test = batch_generator(data, 'test')
        #for (test_images, test_labels) in test:
            #test_accuracy_temp, _ = sess.run([accuracy, test_summary], feed_dict={x_image: test_images, y_: test_labels})

            #batch_count += 1
            #test_accuracy += test_accuracy_temp

        #test_accuracy = test_accuracy / batch_count
        print('test set: accuracy on test set: %0.3f' % test_accuracy)
        print('model saved to ' + checkpoint_path)

        train_writer.close()
        validation_writer.close()

if __name__ == '__main__':
    tf.app.run(main=main)
