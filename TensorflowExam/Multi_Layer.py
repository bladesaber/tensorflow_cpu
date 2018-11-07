import tensorflow as tf
import numpy as np
import time

def variable_with_weight_loss(shape,stddev,mean,wl):
    var = tf.Variable(initial_value=tf.truncated_normal(shape=shape,stddev=stddev,mean=mean))
    # wl 可能是为 L2正则项 分配的权重
    if wl is not None:
        # tf.multiply 是点乘
        weight_loss = tf.multiply(tf.nn.l2_loss(var),wl,name='weight_loss')
        # tf.add_to_collection：把变量放入一个集合，把很多变量变成一个列表
        tf.add_to_collection(name='losses',value=weight_loss)
    return var

batch_size = 32
image_holder = tf.placeholder(dtype=tf.float32,shape=[batch_size,24,24,3])
label_holder = tf.placeholder(dtype=tf.int32,shape=[batch_size])

weight1 = variable_with_weight_loss(shape=[5,5,3,64],stddev=0.1,mean=0.0,wl=0.0)
bias1 = tf.Variable(initial_value=tf.zeros(shape=[64]))
conv1 = tf.nn.relu(tf.nn.conv2d(input=image_holder,filter=weight1,strides=[1,1,1,1],padding='SAME') + bias1)
pool1 = tf.nn.max_pool(input=conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')

weight2 = variable_with_weight_loss(shape=[5,5,64,64],stddev=0.1,mean=0,wl=0.0)
bias2 = tf.Variable(initial_value=tf.constant(value=0.1,shape=[64]))
conv2 = tf.nn.relu(tf.nn.conv2d(input=pool1,filter=weight2,strides=[1,1,1,1],padding='SAME') + bias2)
pool2 = tf.nn.max_pool(input=conv2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')

reshape = tf.reshape(tensor=pool2,shape=[batch_size,-1])
dim = reshape.get_shape()[1].value
weight3 = variable_with_weight_loss(shape=[dim,384],stddev=0.01,mean=0.0,wl=0.004)
bias3 = tf.Variable(initial_value=tf.constant(0.1,shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape,weight3) + bias3)

weight4 = variable_with_weight_loss(shape=[384,192],stddev=0.04,wl=0.004,mean=0.0)
bias4 = tf.Variable(initial_value=tf.constant(0.1,shape=[192]))
local4 = tf.nn.relu(tf.matmul(local3,weight4) + bias4)

weight5 = variable_with_weight_loss(shape=[192,10],stddev=np.sqrt(1.0/192),mean=0.0,wl=0.0)
local5 = tf.nn.softmax(tf.matmul(local4,weight4))

def loss(logits,label):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(label * tf.log(logits),reduction_indices=[1]),name='cross_entropy')
    tf.add_to_collection('lossed',cross_entropy)
    return tf.add_n(inputs=tf.get_collection('lossed'),name='total_loss')

total_loss = loss(logits=local5,label=label_holder)
train_step = tf.train.AdamOptimizer(0.03).minimize(total_loss)

def run(max_step,input,label):
    with tf.Session() as sess:

        init = tf.global_variables_initializer()
        sess.run(init)

        for step in range(max_step):
            start_time = time.time()
            _,loss = sess.run(fetches=[train_step,total_loss],feed_dict={
                image_holder:input,
                label_holder:label
            })
            duration = time.time() - start_time
            if step % 10 ==0:
                example_per_sec = batch_size/duration
                print('example_per_sec: ',example_per_sec,' loss: ',loss)

'''  # Conv Layer Example
#-------------------------------    Model
def weight_variable(shape):
    return tf.Variable(initial_value=tf.truncated_normal(shape=shape,mean=0.0,stddev=0.1))

def bias_variable(shape):
    return tf.Variable(initial_value=tf.truncated_normal(shape=shape,mean=0.0,stddev=0.1))

# strides = [batch, height, width, channels]
# ksize = [batch, height, width, channels]
def conv2d(x,w):
    return tf.nn.conv2d(input=x,filter=w,strides=[1,1,1,1],padding='SAME')

def max_pool(x,size):
    return tf.nn.max_pool(input=x,ksize=[1,size,size,1],strides=[1,2,2,1],padding='VALID')

x = tf.placeholder(dtype=tf.float32,shape=[None,784])
y_ = tf.placeholder(dtype=tf.float32,shape=[None,10])

# Channel_Last [batch_size,height,weight,channels]
x_image = tf.reshape(tensor=x,shape=[-1,28,28,1])

w_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x=x_image,w=w_conv1) + b_conv1)
h_pool1 = max_pool(x=h_conv1,size=2)

w_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(x=h_pool1,w=w_conv2) + b_conv2)
h_pool2 = max_pool(x=h_conv2,size=2)

w_fc1 = weight_variable(shape=[7*7*64,1024])
b_fc1 = bias_variable(shape=[1024])
h_pool2_flat = tf.reshape(tensor=h_pool2,shape=[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)

keep_prob = tf.placeholder(dtype=tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob=keep_prob)

w_fc2 = weight_variable(shape=[1024,10])
b_fc2 = bias_variable(shape=[10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1,w_fc2) + b_fc2)

#--------------------------------------------- Train & Loss
crross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv),reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(0.03).minimize(crross_entropy)

correct_prediction = tf.equal(tf.arg_max(y_conv,1),tf.arg_max(y_,1))
accuracy = tf.reduce_sum(tf.cast(correct_prediction,tf.float32))
'''

'''   # Easy Dense Layer Example
in_units = 784
h1_units = 300
#----------------------   Model
w1 = tf.Variable(
    initial_value=tf.truncated_normal(shape=[in_units,h1_units],
                                      mean=0.0,stddev=np.sqrt(1.0/h1_units),
                                      dtype=tf.float32))
b1 = tf.Variable(initial_value=tf.zeros(shape=[h1_units],dtype=tf.float32))
w2 = tf.Variable(
    initial_value=tf.truncated_normal(shape=[h1_units,10],
                                      mean=0,stddev=np.sqrt(1.0/10),
                                      dtype=tf.float32))
b2 = tf.Variable(initial_value=tf.zeros(shape=[10]))

x = tf.placeholder(dtype=tf.float32,shape=[None,in_units])
keep_prob = tf.placeholder(dtype=tf.float32)

hidden1 = tf.nn.relu(tf.matmul(x,w1)+b1)
hidden1_drop = tf.nn.dropout(hidden1,keep_prob=keep_prob)
y = tf.nn.softmax(tf.matmul(hidden1_drop,w2)+b2)

#-----------------------------------   Train & Loss
y_ = tf.placeholder(shape=[None,10])
cross_entroy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
trans_step = tf.train.AdamOptimizer(0.03).minimize(cross_entroy)

#----------------------------------    Init
init = tf.global_variables_initializer()
'''


