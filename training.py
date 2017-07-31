# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 08:12:02 2017

@author: qxd
"""

import os
import numpy as np
import tensorflow as tf
import data_preparation
import model_generation

N_CLASSES = 2
IM_W = 256
IM_H = 256
BATCH_SIZE = 32
CAPACITY = 10000
MAX_STEP = 15000
LEARNING_RATE = 0.0001

def run_training():
    cat_dir = 'D:\\machine_learning_python\\tensorflow_classification\\PetImages\\Cat\\'
    dog_dir = 'D:\\machine_learning_python\\tensorflow_classification\\PetImages\\Dog\\'
    logs_train_dir = 'D:\\machine_learning_python\\tensorflow_classification\\records\\'
    
    train_images, train_labels, eval_images, eval_labels = data_preparation.get_data(cat_dir, dog_dir)
    train_batch, train_label_batch = data_preparation.create_batch(train_images,
                                                                   train_labels,
                                                                   IM_W,
                                                                   IM_H,
                                                                   BATCH_SIZE,
                                                                   CAPACITY)
    train_logits = model_generation.inference(train_batch, BATCH_SIZE, N_CLASSES)
    train_loss = model_generation.loss(train_logits, train_label_batch)
    train_op = model_generation.training(train_loss, LEARNING_RATE)
    train_acc = model_generation.evaluation_batch(train_logits, train_label_batch)
    
    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()
    
    sess.run(tf.global_variables_initializer())
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print('number of threads is %d' % (len(threads)))
    
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])
            
            if step % 10 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)
                
            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                
    except tf.errors.OutOfRangeError:
        print('Training is done!')
    finally:
        coord.request_stop()
        
    coord.join(threads)
    sess.close()
    
    
    
    
    