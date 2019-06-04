import os  
import tensorflow as tf  
from PIL import Image  
  
orig_picture = 'F:/url_picture/train_test/generate_sample/' 
gen_picture = 'F:/url_picture/Re_train/inputdata/'
 
classes = {'hao','huai'} 


num_samples = 158 
   

def create_record():  
    writer = tf.python_io.TFRecordWriter("dog_train.tfrecords")  
    for index, name in enumerate(classes):  
        class_path = orig_picture +"/"+ name+"/"  
        for img_name in os.listdir(class_path):  
            img_path = class_path + img_name  
            img = Image.open(img_path)  
            img = img.resize((64, 64),Image.ANTIALIAS)  
            img = img.convert('RGB')			
            img_raw = img.tobytes()        
            print (index,img_raw)  
            example = tf.train.Example(  
               features=tf.train.Features(feature={  
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),  
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))  
               }))  
            writer.write(example.SerializeToString())  
    writer.close()  
    

def read_and_decode(filename):  
  
    filename_queue = tf.train.string_input_producer([filename])  

    reader = tf.TFRecordReader()  

    _, serialized_example = reader.read(filename_queue)  


    features = tf.parse_single_example(  
        serialized_example,  
        features={  
            'label': tf.FixedLenFeature([], tf.int64),  
            'img_raw': tf.FixedLenFeature([], tf.string)  
        })  
    label = features['label']  
    img = features['img_raw']  
    img = tf.decode_raw(img, tf.uint8)  
    img = tf.reshape(img, [64, 64, 3])  
    #img = tf.cast(img, tf.float32) * (1. / 255) - 0.5  
    label = tf.cast(label, tf.int32)  
    return img, label  
 
if __name__ == '__main__':  
    create_record()  
    batch = read_and_decode('dog_train.tfrecords')  
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())  
      
    with tf.Session() as sess:  
        sess.run(init_op)    
        coord=tf.train.Coordinator()    
        threads= tf.train.start_queue_runners(coord=coord)  
        
        for i in range(num_samples):    
            example, lab = sess.run(batch)    
            img=Image.fromarray(example, 'RGB')
            img.save(gen_picture+'/'+str(i)+'samples'+str(lab)+'.jpg')   
            print(example, lab)    
        coord.request_stop()    
        coord.join(threads)   
        sess.close()  
