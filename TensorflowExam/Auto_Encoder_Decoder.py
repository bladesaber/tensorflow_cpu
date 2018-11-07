import tensorflow as tf
import numpy as np

class Additive_GaussianNoise_AutoEncoder(object):
    def __init__(self,
                 n_input,
                 n_hidden,
                 # softplus可以看作是ReLu的平滑
                 activation = tf.nn.softplus,
                 optimizer = tf.train.AdamOptimizer(),
                 scale = 0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.activation = activation
        self.scale = tf.placeholder(dtype=tf.float32)
        network_weights = self._initialize_weights()
        self.weights = network_weights

        self.x = tf.placeholder(dtype=tf.float32,shape=[None,self.n_input])
        self.hidden = self.activation(
            # scale*tf.random_normal(shape=(n_input,),dtype=tf.float32) 添加随机噪声
            tf.matmul(self.x + scale*tf.random_normal(shape=(n_input,),dtype=tf.float32),self.weights['w1']) + self.weights['b1']
        )

        self.reconstruction = tf.matmul(self.hidden,self.weights['w2']) + self.weights['b2']
        self.cost = tf.reduce_sum(tf.pow(self.reconstruction-self.x),2.0) * 0.5
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

    def partial_fit(self,X):
        with tf.Session() as sess:
            self.optimizer.run(feed_dict={
                                     self.x:X
                                 })

    def calc_total_cost(self,X):
        with tf.Session() as sess:
            return sess.run(fetches=self.cost,
                     feed_dict={
                         self.x:X
                     })

    def transform(self,X):
        with tf.Session() as sess:
            return sess.run(fetches=self.hidden,
                            feed_dict={
                                self.x:X
                            })

    def generate(self,hidden):
        with tf.Session() as sess:
            return sess.run(fetches=self.reconstruction,
                            feed_dict={
                                self.hidden:hidden
                            })

    def getWeight(self):
        return self.weights

    def _initialize_weights(self):
        all_weights = dict()

        all_weights['w1'] = tf.Variable(
            initial_value=self.xavier_init(fan_in=self.n_input,fan_out=self.n_hidden))
        all_weights['b1'] = tf.Variable(
            initial_value=tf.zeros(shape=[self.n_hidden],dtype=tf.float32))

        all_weights['w2'] = tf.Variable(
            initial_value=self.xavier_init(fan_in=self.n_hidden,fan_out=self.n_input))
        all_weights['b2'] = tf.Variable(
            initial_value=tf.zeros(shape=[self.n_input],dtype=tf.float32))

        return all_weights

    def xavier_init(self,fan_in, fan_out, constant=1):
        low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
        high = constant * np.sqrt(6.0 / (fan_out + fan_in))
        return tf.random_uniform((fan_in, fan_out),
                                 minval=low, maxval=high, dtype='float32')
