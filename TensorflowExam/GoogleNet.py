import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

def inception_arg_scope(weight_decay = 0.00004,
                       stddev = 0.1,
                       batch_norm_var_collection = 'moving_vars'):
    batch_norm_params = {
        'decay':0.9997,
        'epsilon':0.001,
        'update_collections':tf.GraphKeys.UPDATE_OPS,
        'variables_collections':{
            'beta':None,
            'gamma':None,
            'moving_mean':[batch_norm_var_collection],
            'moving_variance':[batch_norm_var_collection]
        }
    }

    with slim.arg_scope([slim.conv2d,slim.fully_connected],weights_regularizer = slim.l2_regularizer(weight_decay)):
        with slim.arg_scope(
            [slim.conv2d],
            weights_initializer = tf.truncated_normal_initializer(stddev=stddev),
            activation_fn = tf.nn.relu,
            normalizer_fn = slim.batch_norm,
            normalizer_params = batch_norm_params ) as sc:
            return sc

def inception_base(inputs,scope = None):
    end_points = {}
    with tf.variable_scope(scope,default_name='inception',values=[inputs]):
        with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],
                            stride = 1,
                            padding = 'VALID'):
            net = slim.conv2d(inputs=inputs,
                              num_outputs=32,
                              kernel_size=[3,3],
                              stride=2,scope='Conv2d_1a_3x3')
            net = slim.conv2d(inputs=net,
                              num_outputs=32,
                              kernel_size=[3,3],
                              scope='Conve2d_2a_3x3')
            net = slim.conv2d(inputs=net,
                              num_outputs=64,
                              kernel_size=[3,3],
                              padding='SAME',scope='Conv2d_2b_3x3')
            net = slim.max_pool2d(inputs=net,
                                  kernel_size=[3,3],
                                  stride=2,scope='MaxPool_3a_3x3')
            net = slim.conv2d(inputs=net,
                              num_outputs=80,
                              kernel_size=[1,1],scope='Conv2d_3b_1x1')
            net = slim.conv2d(inputs=net,
                              num_outputs=192,
                              kernel_size=[3,3],scope='Conve2d_4a_3x3')
            net = slim.max_pool2d(inputs=net,
                                  kernel_size=[3,3],
                                  stride=2,scope='MaxPool_5a_3x3')

    with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],
                        stride = 1,padding = 'SAME'):
        with tf.variable_scope('Mixed_5b'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(inputs=net,
                                       num_outputs=64,
                                       kernel_size=[1,1],scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(inputs=net,
                                       num_outputs=48,
                                       kernel_size=[1,1],scope='Conv2d_0a_1x1')
                branch_1 = slim.conv2d(inputs=branch_1,
                                       num_outputs=64,
                                       kernel_size=[5,5],scope='Conv2d_0b_5x5')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(inputs=net,
                                       num_outputs=64,
                                       kernel_size=[1,1],scope='Conv2d_0a_1x1')
                branch_2 = slim.conv2d(inputs=branch_2,
                                       num_outputs=96,
                                       kernel_size=[3,3],scope='Conv2d_0b_3x3')
                branch_2 = slim.conv2d(inputs=branch_2,
                                       num_outputs=96,
                                       kernel_size=[3,3],scope='Conv2d_0c_3x3')
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(inputs=net,
                                           kernel_size=[3,3],
                                           scope='AvgPool_0a_3x3')
                branch_3 = slim.conv2d(inputs=branch_3,
                                       num_outputs=32,
                                       kernel_size=[1,1],scope='Conv2d_0b_1x1')
            net = tf.concat([branch_0,branch_1,branch_2,branch_3],axis=3)

    with tf.variable_scope(name_or_scope='Mixed_5c'):
        with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(inputs=net,
                                   num_outputs=64,
                                   kernel_size=[1,1],scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(inputs=net,
                                   num_outputs=48,
                                   kernel_size=[1,1],scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(inputs=branch_1,
                                   num_outputs=64,
                                   kernel_size=[5,5],scope='Conv_1_0c_5x5')
        with tf.variable_scope('Branch2'):
            branch_2 = slim.conv2d(inputs=net,
                                   num_outputs=64,
                                   kernel_size=[1,1],scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(inputs=branch_2,
                                   num_outputs=96,
                                   kernel_size=[3,3],scope='Conv2d_0b_3x3')
            branch_2 = slim.conv2d(inputs=branch_2,
                                   num_outputs=96,
                                   kernel_size=[3,3],scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
            branch_3 = slim.avg_pool2d(inputs=net,
                                       kernel_size=[3,3],scope='AvgPool_0a_3x3')
            branch_3 = slim.conv2d(inputs=branch_3,
                                   num_outputs=64,
                                   kernel_size=[1,1],scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)

    with tf.variable_scope('Mixed_5d'):
        with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(inputs=net,
                                   num_outputs=64,
                                   kernel_size=[1,1],scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(inputs=net,
                                   num_outputs=48,
                                   kernel_size=[1,1],scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(inputs=branch_1,
                                   num_outputs=64,
                                   kernel_size=[5,5],scope='Conv2d_0b_5x5')
        with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(inputs=net,
                                   num_outputs=64,
                                   kernel_size=[1,1],scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(inputs=branch_2,
                                   num_outputs=96,
                                   kernel_size=[3,3],scope='Conv2d_0b_3x3')
            branch_2 = slim.conv2d(inputs=branch_2,
                                   num_outputs=96,
                                   kernel_size=[3,3],scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
            branch_3 = slim.avg_pool2d(inputs=net,
                                       kernel_size=[3,3],scope='AvgPool_0a_3x3')
            branch_3 = slim.conv2d(inputs=branch_3,
                                   num_outputs=64,
                                   kernel_size=[1,1],scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)

    with tf.variable_scope('Mixed_6a'):
        with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(inputs=net,
                                   num_outputs=384,
                                   kernel_size=[3,3],
                                   stride=2,
                                   padding='VALID',scope='Conv2d_1a_1x1')
        with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(inputs=net,
                                   num_outputs=64,
                                   kernel_size=[1,1],scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(inputs=branch_1,
                                   num_outputs=96,
                                   kernel_size=[3,3],scope='Conv2d_0b_3x3')
            branch_1 = slim.conv2d(inputs=branch_1,
                                   num_outputs=96,
                                   kernel_size=[3,3],
                                   stride=2,
                                   padding='VALID',scope='Conv2d_1a_1x1')
        with tf.variable_scope('Branch_2'):
            branch_2 = slim.max_pool2d(inputs=net,
                                       kernel_size=[3,3],
                                       stride=2,
                                       padding='VALID',scope='MaxPool_1a_3x3')
        net = tf.concat([branch_0,branch_1,branch_2],3)

    with tf.variable_scope('Mixed_6b'):
        with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(inputs=net,
                                   num_outputs=192,
                                   kernel_size=[1,1],scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(inputs=net,
                                   num_outputs=128,
                                   kernel_size=[1,1],
                                   scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(inputs=branch_1,
                                   num_outputs=128,
                                   kernel_size=[1,7],scope='Conv2d_0b_1x7')
            branch_1 = slim.conv2d(inputs=branch_1,
                                   num_outputs=192,
                                   kernel_size=[7,1],scope='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(inputs=net,
                                   num_outputs=128,
                                   kernel_size=[1,1],scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(inputs=branch_2,
                                   num_outputs=128,
                                   kernel_size=[7,1],scope='Conv2d_0b_7x1')
            branch_2 = slim.conv2d(inputs=branch_2,
                                   num_outputs=128,
                                   kernel_size=[1,7],scope='Conv2d_0c_1x7')
            branch_2 = slim.conv2d(inputs=branch_2,
                                   num_outputs=128,
                                   kernel_size=[7, 1], scope='Conv2d_0d_7x1')
            branch_2 = slim.conv2d(inputs=branch_2,
                                   num_outputs=192,
                                   kernel_size=[1, 7], scope='Conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
            branch_3 = slim.avg_pool2d(inputs=net,
                                       kernel_size=[3,3],scope='AvgPool_0a_3x3')
            branch_3 = slim.conv2d(inputs=branch_3,
                                   num_outputs=192,
                                   kernel_size=[1,1],scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)

    with tf.variable_scope('Mixed_6c'):
        with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(inputs=net,
                                   num_outputs=192,
                                   kernel_size=[1,1],scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(inputs=net,
                                   num_outputs=160,
                                   kernel_size=[1,1],scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(inputs=branch_1,
                                   num_outputs=160,
                                   kernel_size=[1,7],scope='Conv2d_0b_1x7')
            branch_1 = slim.conv2d(inputs=branch_1,
                                   num_outputs=192,
                                   kernel_size=[7,1],scope='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(inputs=net,
                                   num_outputs=160,
                                   kernel_size=[1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(inputs=branch_2,
                                   num_outputs=160,
                                   kernel_size=[7, 1], scope='Conv2d_0b_7x1')
            branch_2 = slim.conv2d(inputs=branch_2,
                                   num_outputs=160,
                                   kernel_size=[1, 7], scope='Conv2d_0c_1x7')
            branch_2 = slim.conv2d(inputs=branch_2,
                                   num_outputs=160,
                                   kernel_size=[7, 1], scope='Conv2d_0d_7x1')
            branch_2 = slim.conv2d(inputs=branch_2,
                                   num_outputs=192,
                                   kernel_size=[1, 7], scope='Conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
            branch_3 = slim.avg_pool2d(inputs=net,
                                       kernel_size=[3,3],scope='AvgPool_0a_3x3')
            branch_3 = slim.conv2d(inputs=branch_3,
                                   num_outputs=192,
                                   kernel_size=[1,1],scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)

    with tf.variable_scope('Mixed_6d'):
        with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(inputs=net,
                                   num_outputs=192,
                                   kernel_size=[1,1],scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(inputs=net,
                                   num_outputs=160,
                                   kernel_size=[1,1],scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(inputs=branch_1,
                                   num_outputs=160,
                                   kernel_size=[1,7],scope='Conv2d_0b_1x7')
            branch_1 = slim.conv2d(inputs=branch_1,
                                   num_outputs=192,
                                   kernel_size=[7,1],scope='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(inputs=net,
                                   num_outputs=160,
                                   kernel_size=[1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(inputs=branch_2,
                                   num_outputs=160,
                                   kernel_size=[7, 1], scope='Conv2d_0b_7x1')
            branch_2 = slim.conv2d(inputs=branch_2,
                                   num_outputs=160,
                                   kernel_size=[1, 7], scope='Conv2d_0c_1x7')
            branch_2 = slim.conv2d(inputs=branch_2,
                                   num_outputs=160,
                                   kernel_size=[7, 1], scope='Conv2d_0d_7x1')
            branch_2 = slim.conv2d(inputs=branch_2,
                                   num_outputs=192,
                                   kernel_size=[1, 7], scope='Conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
            branch_3 = slim.avg_pool2d(inputs=net,
                                       kernel_size=[3,3],scope='AvgPool_0a_3x3')
            branch_3 = slim.conv2d(inputs=branch_3,
                                   num_outputs=192,
                                   kernel_size=[1,1],scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)

    with tf.variable_scope('Mixed_6e'):
        with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(inputs=net,
                                   num_outputs=192,
                                   kernel_size=[1,1],scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(inputs=net,
                                   num_outputs=192,
                                   kernel_size=[1,1],scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(inputs=branch_1,
                                   num_outputs=192,
                                   kernel_size=[1,7],scope='Conv2d_0b_1x7')
            branch_1 = slim.conv2d(inputs=branch_1,
                                   num_outputs=192,
                                   kernel_size=[7,1],scope='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(inputs=net,
                                   num_outputs=192,
                                   kernel_size=[1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(inputs=branch_2,
                                   num_outputs=192,
                                   kernel_size=[7, 1], scope='Conv2d_0b_7x1')
            branch_2 = slim.conv2d(inputs=branch_2,
                                   num_outputs=192,
                                   kernel_size=[1, 7], scope='Conv2d_0c_1x7')
            branch_2 = slim.conv2d(inputs=branch_2,
                                   num_outputs=160,
                                   kernel_size=[7, 1], scope='Conv2d_0d_7x1')
            branch_2 = slim.conv2d(inputs=branch_2,
                                   num_outputs=192,
                                   kernel_size=[1, 7], scope='Conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
            branch_3 = slim.avg_pool2d(inputs=net,
                                       kernel_size=[3,3],scope='AvgPool_0a_3x3')
            branch_3 = slim.conv2d(inputs=branch_3,
                                   num_outputs=192,
                                   kernel_size=[1,1],scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)

    end_points['Mixed_6e'] = net

    with tf.variable_scope('Mixed_7a'):
        with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(inputs=net,
                                   num_outputs=192,
                                   kernel_size=[1, 1], scope='Conv2d_0a_1x1')
            branch_0 = slim.conv2d(inputs=branch_0,
                                   num_outputs=320,
                                   kernel_size=[3,3],
                                   stride=2,
                                   padding='VALID',scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(inputs=net,
                                   num_outputs=192,
                                   kernel_size=[1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(inputs=branch_1,
                                   num_outputs=192,
                                   kernel_size=[1, 7], scope='Conv2d_0b_1x7')
            branch_1 = slim.conv2d(inputs=branch_1,
                                   num_outputs=192,
                                   kernel_size=[7, 1], scope='Conv2d_0c_7x1')
            branch_1 = slim.conv2d(inputs=branch_1,
                                   num_outputs=192,
                                   kernel_size=[3,3],
                                   stride=2,
                                   padding='VALID',scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_2'):
            branch_2 = slim.max_pool2d(inputs=net,
                                       kernel_size=[3, 3],
                                       stride=2,
                                       padding='VALID',scope='Conv2d_0a_1x1')
        net = tf.concat([branch_0, branch_1, branch_2], 3)

    with tf.variable_scope('Mixed_7b'):
        with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(inputs=net,
                                   num_outputs=320,
                                   kernel_size=[1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(inputs=net,
                                   num_outputs=384,
                                   kernel_size=[1, 1], scope='Conv2d_0a_1x1')
            branch_1 = tf.concat([
                slim.conv2d(inputs=branch_1,
                            num_outputs=384,
                            kernel_size=[1, 3], scope='Conv2d_0b_1x3'),
                slim.conv2d(inputs=branch_1,
                            num_outputs=384,
                            kernel_size=[3, 1], scope='Conv2d_0c_3x1')
            ],3)
        with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(inputs=net,
                                   num_outputs=448,
                                   kernel_size=[1, 1],scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(inputs=net,
                                   num_outputs=384,
                                   kernel_size=[3, 3], scope='Conv2d_0b_3x3')
            branch_2 = tf.concat([
                slim.conv2d(inputs=branch_1,
                            num_outputs=384,
                            kernel_size=[1, 3], scope='Conv2d_0c_1x3'),
                slim.conv2d(inputs=branch_1,
                            num_outputs=384,
                            kernel_size=[3, 1], scope='Conv2d_0d_3x1')
            ],3)
        with tf.variable_scope('Branch_3'):
            branch_3 = slim.avg_pool2d(inputs=net,
                                       kernel_size=[3,3],scope='AvgPool_0a_3x3')
            branch_3 = slim.conv2d(inputs=branch_3,
                                   num_outputs=192,
                                   kernel_size=[1,1],scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0, branch_1, branch_2,branch_2], 3)

    with tf.variable_scope('Mixed_7b'):
        with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(inputs=net,
                                   num_outputs=320,
                                   kernel_size=[1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(inputs=net,
                                   num_outputs=384,
                                   kernel_size=[1, 1], scope='Conv2d_0a_1x1')
            branch_1 = tf.concat([
                slim.conv2d(inputs=branch_1,
                            num_outputs=384,
                            kernel_size=[1, 3], scope='Conv2d_0b_1x3'),
                slim.conv2d(inputs=branch_1,
                            num_outputs=384,
                            kernel_size=[3, 1], scope='Conv2d_0c_3x1')
            ],3)
        with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(inputs=net,
                                   num_outputs=448,
                                   kernel_size=[1, 1],scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(inputs=net,
                                   num_outputs=384,
                                   kernel_size=[3, 3], scope='Conv2d_0b_3x3')
            branch_2 = tf.concat([
                slim.conv2d(inputs=branch_1,
                            num_outputs=384,
                            kernel_size=[1, 3], scope='Conv2d_0c_1x3'),
                slim.conv2d(inputs=branch_1,
                            num_outputs=384,
                            kernel_size=[3, 1], scope='Conv2d_0d_3x1')
            ],3)
        with tf.variable_scope('Branch_3'):
            branch_3 = slim.avg_pool2d(inputs=net,
                                       kernel_size=[3,3],scope='AvgPool_0a_3x3')
            branch_3 = slim.conv2d(inputs=branch_3,
                                   num_outputs=192,
                                   kernel_size=[1,1],scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0, branch_1, branch_2,branch_2], 3)

    with tf.variable_scope('Mixed_7c'):
        with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(inputs=net,
                                   num_outputs=320,
                                   kernel_size=[1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(inputs=net,
                                   num_outputs=384,
                                   kernel_size=[1, 1], scope='Conv2d_0a_1x1')
            branch_1 = tf.concat([
                slim.conv2d(inputs=branch_1,
                            num_outputs=384,
                            kernel_size=[1, 3], scope='Conv2d_0b_1x3'),
                slim.conv2d(inputs=branch_1,
                            num_outputs=384,
                            kernel_size=[3, 1], scope='Conv2d_0c_3x1')
            ],3)
        with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(inputs=net,
                                   num_outputs=448,
                                   kernel_size=[1, 1],scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(inputs=net,
                                   num_outputs=384,
                                   kernel_size=[3, 3], scope='Conv2d_0b_3x3')
            branch_2 = tf.concat([
                slim.conv2d(inputs=branch_1,
                            num_outputs=384,
                            kernel_size=[1, 3], scope='Conv2d_0c_1x3'),
                slim.conv2d(inputs=branch_1,
                            num_outputs=384,
                            kernel_size=[3, 1], scope='Conv2d_0d_3x1')
            ],3)
        with tf.variable_scope('Branch_3'):
            branch_3 = slim.avg_pool2d(inputs=net,
                                       kernel_size=[3,3],scope='AvgPool_0a_3x3')
            branch_3 = slim.conv2d(inputs=branch_3,
                                   num_outputs=192,
                                   kernel_size=[1,1],scope='Conv2d_0b_1x1')
        net = tf.concat([branch_0, branch_1, branch_2,branch_2], 3)

def inception(inputs,
              num_classes=1000,
              is_training = True,
              dropout_keep_prob = 0.8,
              prediction_fn = tf.nn.softmax,
              spatital_squeeze = True,
              reuse = None,
              scope = 'Inceprion'):
    with tf.variable_scope(name_or_scope=scope,default_name='Inception',
                           values=[inputs,num_classes],reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm,slim.dropout],
                            is_training = is_training ):
            net,end_points = inception_base(inputs=inputs,scope=scope)

        with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],
                            stride = 1,padding = 'SAME'):
            aux_logits = end_points['Mixed_6e']
            with tf.variable_scope('AuxLogits'):
                aux_logits = slim.avg_pool2d(inputs=aux_logits,
                                             kernel_size=[5,5],
                                             stride=3,
                                             padding='VALID',scope='AvgPool_1a_5x5')
                aux_logits = slim.conv2d(inputs=aux_logits,
                                         num_outputs=128,
                                         kernel_size=[1,1],scope='Conv2d_1b_1x1')
                aux_logits = slim.conv2d(inputs=aux_logits,
                                         num_outputs=768,
                                         kernel_size=[5,5],
                                         weights_initializer=tf.truncated_normal(stddev=0.01),
                                         padding='VALID',scope='Conv2d_2a_5x5')
                aux_logits = slim.conv2d(inputs=aux_logits,
                                         num_outputs=num_classes,
                                         kernel_size=[1,1],
                                         activation_fn=None,
                                         normalizer_fn=None,
                                         weights_initializer=tf.truncated_normal(stddev=0.001),scope='Conv2d_2b_1x1')
                if spatital_squeeze:
                    # squeeze函数的作用是去掉维度为1的维
                    aux_logits = tf.squeeze(input=aux_logits,
                                            axis=[1,2],name='SpatialSqueeze')
                end_points['AuxLogits'] = aux_logits

            with tf.variable_scope('Logits'):
                net = slim.avg_pool2d(inputs=net,
                                      kernel_size=[8,8],
                                      padding='VALID',scope='AvgPool_1a_8x8')
                net = slim.dropout(inputs=net,
                                   keep_prob=dropout_keep_prob,scope='Dropout_1b')
                end_points['PreLogits'] = net
                logits = slim.conv2d(inputs=net,
                                         num_outputs=num_classes,
                                         kernel_size=[1,1],
                                         activation_fn=None,
                                         normalizer_fn=None,scope='Conv2d_1c_1x1')
                if spatital_squeeze:
                    logits = tf.squeeze(input=logits,
                                        axis=[2,4],name='SpatialSqueeze')
            end_points['Logits'] = logits
            end_points['Predictions'] = prediction_fn(logits=logits,scope = 'Predictions')
    return logits,end_points

batch_size = 32
height,width = 299,299
inputs = tf.random_uniform(shape=(batch_size,height,width,3))
with slim.arg_scope(inception_arg_scope()):
    logits,end_points = inception(inputs,is_training=False)

init = tf.global_variables_initializer
with tf.Session() as sess:
    sess.run(init)


