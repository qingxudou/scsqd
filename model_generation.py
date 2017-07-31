# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 06:01:19 2017

@author: qxd
"""

import tensorflow as tf

def inference(images, batch_size, n_classes):
    '''Generate the model
    Args:
        images: input batches of images
        batch_size: nuumber of images in a batch
        n_classes: number of classes (here is 2)
    Return:
        tensor of computed logits, [batch_size, n_classes]
    '''
    #conv1 
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weight', 
                                  shape = [5,5,3,16],
                                  dtype = tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.03,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1,1,1,1,], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name='conv1')
    #pool1
    with tf.variable_scope('pooling1') as scope:
         pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1],strides=[1,2,2,1],
                                padding='SAME', name='pooling1')
         norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                           beta=0.75,name='norm1')
         
    #conv2 
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weight', 
                                  shape = [5,5,16,16],
                                  dtype = tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.03,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1,1,1,1,], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')
    #pool2
    with tf.variable_scope('pooling2') as scope:
         norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                           beta=0.75,name='norm2')
         pool2 = tf.nn.max_pool(norm2, ksize=[1,3,3,1],strides=[1,2,2,1],
                                padding='SAME', name='pooling2')
    #fconn1
    with tf.variable_scope('fconn1') as scope:
        reshape = tf.reshape(pool2, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights',
                                  shape=[dim, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.03,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        fconn1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    #fconn2
    with tf.variable_scope('fconn2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[128, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.003,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        fconn2 = tf.nn.relu(tf.matmul(fconn1, weights) + biases, name='fconn2')
    #softmax
    with tf.variable_scope('softmax_liner') as scope:
        weights = tf.get_variable('softmax_linear',
                                  shape=[128, n_classes],
                                  dtype = tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.003,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(fconn2, weights), biases, name='softmax_linear')
    return softmax_linear

def loss(logits, labels):
    '''compute loss from logits and labels
    '''
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='xentropy')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name+'/loss', loss)
    return loss

def training(loss, learning_rate):
    '''
    Training ops, the return by this function is what passed to sess.run()
    '''
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluation_batch(logits, labels):
    '''
    Evaluate the quality of the logits at predicting the label
    '''
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name+'/accuracy', accuracy)
    return accuracy


                              