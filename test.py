# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 19:04:46 2018

@author: sys2009
"""
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import model
from input_data import get_files

#获取一张照片
def get_one_image(train):
    #输入参数：train，训练图片的路径
    #返回参数：image，从训练图片中随机抽取一张图片
    n =len(train)
    ind = np.random.randint(0,n)
    img_dir = train[ind] #随机选取测试的图片
    
    img = Image.open(img_dir)
    plt.imshow(img)
    img.show()
    imag = img.resize([64,64])
    image = np.array(imag)
    return image
    
    #测试图片
def evaluate_one_image(image_array):
    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 2
        
        image = tf.cast(image_array,tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image,[1,64,64,3])
        
        logit = model.inference(image,BATCH_SIZE,N_CLASSES)
        logit = tf.nn.softmax(logit)
        
        x = tf.placeholder(tf.float32,shape=[64,64,3])
        
        logs_train_dir = 'F:/url_picture/Re_train/inputdata'
        
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            print("Reading checkpoint...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess,ckpt.model_checkpoint_path)
                print('loading success,global_step is %s'% global_step)
            else:
                print('No checkpoint file found')
                
                
            prediction = sess.run(logit,feed_dict={x: image_array})
            print(prediction)
            max_index = np.argmax(prediction)
            if max_index == 0:
                print('This is a hao with possibility %.6f'%prediction[:,0])
            elif max_index == 1:
                print('This is a huai with possibility %.6f'%prediction[:,1])    
            
                
                
if __name__ == '__main__':
    train_dir = 'F:/url_picture/Re_train/inputdata/test'
    
    train,train_label,val,val_label = get_files(train_dir,0.3)
    img = get_one_image(val)
    evaluate_one_image(img)               
