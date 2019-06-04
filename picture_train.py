import os
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
 
train_dir = 'F:/url_picture/Re_train/inputdata'
 
niao = []
label_niao = []
gou = []
label_gou = []
mao = []
label_mao = []
tu = []
label_tu = []
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
 


    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 32, 
                                                capacity = capacity)
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch  
          
#=========================================================================
import tensorflow as tf

def inference(images, batch_size, n_classes):

    with tf.variable_scope('conv1') as scope:
        
        weights = tf.Variable(tf.truncated_normal(shape=[3,3,3,64], stddev = 1.0, dtype = tf.float32), 
                              name = 'weights', dtype = tf.float32)
        
        biases = tf.Variable(tf.constant(value = 0.1, dtype = tf.float32, shape = [64]),
                             name = 'biases', dtype = tf.float32)
        
        conv = tf.nn.conv2d(images, weights, strides=[1,1,1,1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name= scope.name)
        

    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME', name='pooling1')
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm1')
 

    with tf.variable_scope('conv2') as scope:
        weights = tf.Variable(tf.truncated_normal(shape=[3,3,64,16], stddev = 0.1, dtype = tf.float32), 
                              name = 'weights', dtype = tf.float32)
        
        biases = tf.Variable(tf.constant(value = 0.1, dtype = tf.float32, shape = [16]),
                             name = 'biases', dtype = tf.float32)
        
        conv = tf.nn.conv2d(norm1, weights, strides = [1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')
 

    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0,beta=0.75,name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1,3,3,1], strides=[1,1,1,1],padding='SAME',name='pooling2')
 

    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.Variable(tf.truncated_normal(shape=[dim,128], stddev = 0.005, dtype = tf.float32),
                             name = 'weights', dtype = tf.float32)
        
        biases = tf.Variable(tf.constant(value = 0.1, dtype = tf.float32, shape = [128]), 
                             name = 'biases', dtype=tf.float32)
        
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        

    with tf.variable_scope('local4') as scope:
        weights = tf.Variable(tf.truncated_normal(shape=[128,128], stddev = 0.005, dtype = tf.float32),
                              name = 'weights',dtype = tf.float32)
        
        biases = tf.Variable(tf.constant(value = 0.1, dtype = tf.float32, shape = [128]),
                             name = 'biases', dtype = tf.float32)
        
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')
 

            
      
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.Variable(tf.truncated_normal(shape=[128, n_classes], stddev = 0.005, dtype = tf.float32),
                              name = 'softmax_linear', dtype = tf.float32)
        
        biases = tf.Variable(tf.constant(value = 0.1, dtype = tf.float32, shape = [n_classes]),
                             name = 'biases', dtype = tf.float32)
        
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')
 
    return softmax_linear
 

def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy =tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name+'/loss', loss)
    return loss
 
#--------------------------------------------------------------------------

def trainning(loss, learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step= global_step)
    return train_op
 

def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name+'/accuracy', accuracy)
    return accuracy



import os
import numpy as np
import tensorflow as tf
import input_data
import model
 

N_CLASSES = 2 
IMG_W = 64   
IMG_H = 64
BATCH_SIZE = 10
CAPACITY = 200
MAX_STEP = 200 
learning_rate = 0.0001 
 

train_dir = 'F:/url_picture/Re_train/inputdata' 
logs_train_dir = 'F:/url_picture/Re_train/inputdata'  


train, train_label, val, val_label = input_data.get_files(train_dir, 0.3)

train_batch,train_label_batch = input_data.get_batch(train, train_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

val_batch, val_label_batch = input_data.get_batch(val, val_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY) 
 

train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)
train_loss = model.losses(train_logits, train_label_batch)        
train_op = model.trainning(train_loss, learning_rate)
train_acc = model.evaluation(train_logits, train_label_batch)
 

test_logits = model.inference(val_batch, BATCH_SIZE, N_CLASSES)
test_loss = model.losses(test_logits, val_label_batch)        
test_acc = model.evaluation(test_logits, val_label_batch)
 

summary_op = tf.summary.merge_all() 
 

sess = tf.Session()  

train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph) 


saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())  

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
 

try:

    for step in np.arange(MAX_STEP):
        if coord.should_stop():
            break

        _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])
        

        if step % 10  == 0:
            print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))
            summary_str = sess.run(summary_op)
            train_writer.add_summary(summary_str, step)

        if (step + 1) == MAX_STEP:
            checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)
       
except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
 
finally:
    coord.request_stop()
