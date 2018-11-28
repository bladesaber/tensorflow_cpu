import tensorflow as tf
import numpy

def conv_batch_norm(input,n_out,phase_train):
    beta_init = tf.constant_initializer(value=0.0,dtype=tf.float32)
    gamma_init = tf.constant_initializer(value=1.0,dtype=tf.float32)

    beta = tf.get_variable('beta')

def resnet_layer(name,filter_size,kernel_size,strides,input,stddev,mean,batch_normalization = True,activation = False):
    with tf.name_scope(name=name) as scope:
        dim = input.get_shape()[-1].value
        shape = [kernel_size,kernel_size,dim,filter_size]
        kernel = tf.Variable(initial_value=tf.truncated_normal(shape=shape,mean=mean,stddev=stddev),dtype=tf.float32,name='weight')
        bias = tf.Variable(tf.constant(value=0.0,dtype=tf.float32,shape=[filter_size]),name='bias')
        stride = [1,strides,strides,1]
        output = tf.nn.conv2d(input=input,filter=kernel,strides=strides,padding='SAME') + bias



