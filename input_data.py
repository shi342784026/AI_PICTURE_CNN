import os
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
 

 
train_dir = 'F:/url_picture/Re_train/image_data/inputdata'
 
hao = []
label_hao = []
huai = []
label_huai = []


 

def get_files(file_dir, ratio):
    for file in os.listdir(file_dir+'/hao'):
        hao.append(file_dir +'/hao'+'/'+ file) 
        label_hao.append(0)
    for file in os.listdir(file_dir+'/huai'):
        huai.append(file_dir +'/huai'+'/'+file)
        label_huai.append(1)

 

    image_list = np.hstack((hao, huai))
    label_list = np.hstack((label_hao, label_huai))
 
    
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    

    
   
    all_image_list = list(temp[:, 0])
    all_label_list = list(temp[:, 1])
 
    
    n_sample = len(all_label_list)
    n_val = int(math.ceil(n_sample*ratio))   
    n_train = n_sample - n_val   
 
    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]
    val_images = all_image_list[n_train:-1]
    val_labels = all_label_list[n_train:-1]
    val_labels = [int(float(i)) for i in val_labels]
 
    return tra_images, tra_labels, val_images, val_labels
    
    

def get_batch(image, label, image_W, image_H, batch_size, capacity):

    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)
 
    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])
 
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0]) #read img from a queue  
    

    image = tf.image.decode_jpeg(image_contents, channels=3) 
    

    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    image = tf.image.per_image_standardization(image)
 

#image_batch: 4D tensor [batch_size, width, height, 3],dtype=tf.float32 
#label_batch: 1D tensor [batch_size], dtype=tf.int32
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 32, 
                                                capacity = capacity)
    
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch            
