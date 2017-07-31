# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 19:42:55 2017

@author: qxd
"""

import tensorfolw as tf
import data_preparation
import model_generation

N_CLASSES = 2
IM_W = 256
IM_H = 256
BATCH_SIZE = 50
CAPACITY = 10000

def evaluation():
    cat_dir = 'D:\\machine_learning_python\\tensorflow_classification\\PetImages\\Cat\\'
    dog_dir = 'D:\\machine_learning_python\\tensorflow_classification\\PetImages\\Dog\\'
    logs_train_dir = 'D:\\machine_learning_python\\tensorflow_classification\\records\\'
    train_images, train_labels, eval_images, eval_labels = data_preparation.get_data(cat_dir, dog_dir)
    
    iter = int(len(eval_images)/BATCH_SIZE)
    
    eval_batch, eval_label_batch = data_preparation.create_batch(eval_images,
                                                                   eval_labels,
                                                                   IM_W,
                                                                   IM_H,
                                                                   BATCH_SIZE,
                                                                   CAPACITY)
    logits = model_generation.inference(eval_batch, BATCH_SIZE, N_CLASSES)
    accu_batch = model_generation.evaluation_batch(logits, eval_label_batch)
    
    saver = tf.strain.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(logs_train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading success, global_step is $s' % global_step)
        else:
            print('No checkpoint file fouund')
        accuracy = 0.0;
        for i in range(iter):
            accuracy += sess.run(accu_batch)
        accuracy = accuracy / iter
        print('Accuracy is about %.3f' % (accuracy))
