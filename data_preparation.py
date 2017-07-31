# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 12:18:54 2017

@author: qxd
"""

import tensorflow as tf
import numpy as np
import os

img_width = 256;
img_hright = 256;

#cat_dir = 'D:\\machine_learning_python\\tensorflow_classification\\PetImages\\Cat\\'
#dog_dir = 'D:\\machine_learning_python\\tensorflow_classification\\PetImages\\Dog\\'

def get_data(cat_dir, dog_dir):
    '''
    Args:
        cat_dir: directory of cat images
        dog_dir: directory of dog images
    outputs:
        list of images and the related labels
    '''
    cats_train = []
    cats_eval =[]
    cats_train_label = []
    cats_eval_label = []
    dogs_train = []
    dogs_train_label = []
    dogs_eval = []
    dogs_eval_label = []
    count = 0
    for file in os.listdir(cat_dir):
        if count < 10000:
            cats_train.append(cat_dir + file)
            cats_train_label.append(0)
            count += 1
        else:
            cats_eval.append(cat_dir + file)
            cats_eval_label.append(0)
            count += 1
    count = 0
    for file in os.listdir(dog_dir):
        if count < 10000:
            dogs_train.append(dog_dir + file)
            dogs_train_label.append(1)
            count += 1
        else:
            dogs_eval.append(dog_dir + file)
            dogs_eval_label.append(1)
            count += 1
    
            
    train_images = np.hstack((cats_train, dogs_train))
    train_labels = np.hstack((cats_train_label, dogs_train_label)) 
    
    eval_images = np.hstack((cats_eval, dogs_eval))
    eval_labels = np.hstack((cats_eval_label, dogs_eval_label))
    
    temp = np.array([train_images, train_labels])
    temp = temp.transpose()
    np.random.shuffle(temp)
    
    train_images = list(temp[:,0])
    train_labels = list(temp[:,1])
    train_labels = [int(i) for i in train_labels]

    temp = np.array([eval_images, eval_labels])
    temp = temp.transpose()
    np.random.shuffle(temp)
    
    eval_images = list(temp[:,0])
    eval_labels = list(temp[:,1])
    eval_labels = [int(i) for i in eval_labels]
    
    
    return train_images, train_labels, eval_images, eval_labels

def create_batch(im_list, la_list, im_w, im_h, batch_size, capacity):
    '''
    Args:
        im_list: list of images
        la_list: list of labels
        im_w: width of images
        im_h: height of images
        batch_size: number of images in a batch
        capacity: the maximum elements in queue
    Return:
        im_batch: 4D tensor [batch_size, im_w, im_h, 3]. dtype=tf.float32
        la_batch: 1D tensor [batch_size], dtype=tf.int32
    '''
    im_list = tf.cast(im_list, tf.string)
    la_list = tf.cast(la_list, tf.int32)
    
    input_queue = tf.train.slice_input_producer([im_list, la_list])
    label = input_queue[1]
    image_index = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_index, channels=3)  
    
    image = tf.image.resize_image_with_crop_or_pad(image, im_w, im_h)
    image = tf.image.per_image_standardization(image)
    im_batch, la_batch = tf.train.batch([image, label], batch_size=batch_size, 
                                        num_threads=1, capacity=capacity)
    la_batch = tf.reshape(la_batch, [batch_size])
    return im_batch, la_batch

#import matplotlib.pyplot as plt
#
#batch_size = 8
#capacity = 2000
#im_w =256
#im_h =256
#cat_dir = 'D:\\machine_learning_python\\tensorflow_classification\\PetImages\\Cat\\'
#dog_dir = 'D:\\machine_learning_python\\tensorflow_classification\\PetImages\\Dog\\'
#
#im_list, la_list = get_data(cat_dir, dog_dir)
#
#im_batch, la_batch = create_batch(im_list, la_list, im_w, im_h, batch_size, capacity)
#
#with tf.Session() as sess:
#    i = 0
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(coord=coord)
#    try:
#        while not coord.should_stop() and i < 3:
#            im_b, la_b = sess.run([im_batch, la_batch])
#            
#            for j in np.arange(batch_size):
#                print('label: %d' % la_b[j])
#                plt.imshow(im_b[j, :,:,:])
#                plt.show()
#                i+=1
#    except tf.errors.OutOfRangeError:
#        print('done!')
#    finally:
#        coord.request_stop()
#    coord.join(threads)
#    sess.close()
    
    
     
