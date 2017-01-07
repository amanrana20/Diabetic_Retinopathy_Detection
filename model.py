import numpy as np
import random
import cv2, os
import tensorflow as tf


class Tensorflow_Model():
    
    __W = None
    __b = None
    
    def __init__(self, image_dims, output_dims):
        self.dims_image = image_dims
        self.dims_output = output_dims
        self.padding = 'SAME'
        self.sess =  tf.Session()
        
        self.__W = {
                1: tf.Variable(tf.truncated_normal([5, 5, 3, 64], stddev=0.1)),
                2: tf.Variable(tf.truncated_normal([5, 5, 64, 128], stddev=0.1)),
                3: tf.Variable(tf.truncated_normal([5, 5, 128, 256], stddev=0.1)),
                4: tf.Variable(tf.truncated_normal([43*64*256, 1024], stddev=0.1)),
                5: tf.Variable(tf.truncated_normal([1024, output_dims], stddev=0.1)),
            }
        
        self.__b = {
                1: tf.Variable(tf.random_normal([64])),
                2: tf.Variable(tf.random_normal([128])),
                3: tf.Variable(tf.random_normal([256])),
                4: tf.Variable(tf.random_normal([1024])),
                5: tf.Variable(tf.random_normal([output_dims])),
            }

    def model(self, inp):
        # Layer 1
        input = inp
        layer1_conv1 = tf.nn.conv2d(input, self.__W[1], strides=[1, 1, 1, 1], padding=self.padding)
        layer1_relu1 = tf.nn.relu(tf.nn.bias_add(layer1_conv1, self.__b[1]))
        layer1_max_pool1 = tf.nn.max_pool(layer1_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=self.padding)

        # Layer 2
        layer2_conv1 = tf.nn.conv2d(layer1_max_pool1, self.__W[2], strides=[1, 1, 1, 1], padding=self.padding)
        layer2_relu1 = tf.nn.relu(tf.nn.bias_add(layer2_conv1, self.__b[2]))
        layer2_max_pool1 = tf.nn.max_pool(layer2_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=self.padding)
    
        # Layer 3
        layer3_conv1 = tf.nn.conv2d(layer2_max_pool1, self.__W[3], strides=[1, 1, 1, 1], padding=self.padding)
        layer3_relu1 = tf.nn.relu(tf.nn.bias_add(layer3_conv1, self.__b[3]))
        layer3_max_pool1 = tf.nn.max_pool(layer3_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=self.padding)
        w4 = (layer3_max_pool1.get_shape()[1:]).as_list()
#        self.__W[4] = tf.Variable(tf.truncated_normal([tf.reduce_prod(self.w4), 1024], stddev=0.1))
#        print(self.w4)

        # Flatten
        flatten = tf.reshape(layer3_max_pool1, [-1, tf.reduce_prod(w4)])
    
        # Fully Connected Network
        fc1 = tf.nn.relu(tf.matmul(flatten, self.__W[4]) + self.__b[4])
        out = tf.nn.relu(tf.matmul(fc1, self.__W[5]) + self.__b[5])
        print(out.get_shape().as_list())
    
        return out

    def get_x_y(self, data):
        x = data[:, 0]
        print(data[:, 1])
        y = tf.constant(tf.one_hot(tf.transpose(data[:, 1]), data.shape[0]+1, 1., 0., -1))
        print('y shape'.format(np.array(y).shape))
        return np.array(x).reshape([-1, self.dims_image['height'], self.dims_image['width'], self.dims_image['channel']]), y


    def train(self, data):
        with tf.device('/cpu:0'):
            init = tf.initialize_all_variables()
            x = tf.placeholder(tf.float32, [None, self.dims_image['height'], self.dims_image['width'], self.dims_image['channel']])
            y = tf.placeholder(tf.float32, [None, self.dims_output])
            self.sess.run(init)
            _y = self.model(x)
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(_y, y))
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
            corr = tf.equal(tf.argmax(_y, 1), tf.argmax(y, 1))
            accr = tf.reduce_mean(tf.cast(corr, tf.float32))
            
            batch_x, batch_y = self.get_x_y(data)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            avg_cost += sess.run(cost, feed_dict={x: batch_x, y: batch_y})/self.dims_output
            train_acc = sess.run(accr, feed_dict={x: batch_x, y: batch_y})
            print('Average Cost: {}, Training Accuracy: {}'.format(avg_cost, train_acc))
