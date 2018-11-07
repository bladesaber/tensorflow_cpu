import tensorflow as tf
import numpy

'''  #----------------------    Here is a Beginner
b = tf.Variable(initial_value=tf.zeros(shape=[10]),name='b',dtype='float32')
w = tf.Variable(name='w',
                initial_value=tf.random_uniform(shape=[20,10],
                minval=-1,maxval=1),dtype='float32')
x = tf.placeholder(name='x',dtype='float32')
relu = tf.nn.relu(tf.matmul(x,w)+b)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(10):
    input = numpy.matrix(numpy.random.randint(low=0,high=10,size=(20,)))
    result = sess.run(fetches=relu,
                      feed_dict={x:input})
    print('epoch: ',i,'input: ',input)
    print('     result: ',result)
'''
#--------------------------------------------------------------------------------

'''
Example For Mnist Handwritting DataSet
'''

class DataGenner(object):
    def dataSet_mini(self):
        path = 'C:/Users/bladesaber\Desktop/tensorflow_cpu\DataSet\mnist.npz'
        f = numpy.load(path)
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        f.close()
        return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = DataGenner().dataSet_mini()

#-----------------    Model
# 这里的 None 指代 样本数目，tensorflow 默认使用 Channel_Last
x = tf.placeholder(name='x',dtype='float32',shape=[None,784])

w = tf.Variable(initial_value=tf.zeros(shape=[784,30],dtype='float32'))
b = tf.Variable(initial_value=tf.zeros(shape=[30],dtype='float32'))
net1 = tf.sigmoid(tf.matmul(x,w)+b)

w2 = tf.Variable(initial_value=tf.zeros(shape=[30,10],dtype='float32'))
b2 = tf.Variable(initial_value=tf.zeros(shape=[10],dtype='float32'))
y = tf.nn.softmax(tf.matmul(net1,w2)+b2)

y_ = tf.placeholder(dtype='float32',shape=[None,10])
# 虽然很奇怪，但 reduce_mean 只是求均值
# reduction_indices 为缩减维度，待考究
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),
                                              reduction_indices=[1]))

correct_prediction = tf.equal(tf.arg_max(input=y,dimension=1),tf.arg_max(input=y_,dimension=1))
correct_ration = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#-----------------   Train & Loss
train_step = tf.train.GradientDescentOptimizer(0.035).minimize(cross_entropy)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train/255
x_test = x_test/255

with tf.Session() as sess:

    # -----------------   Init
    init = tf.global_variables_initializer()
    sess.run(init)

    correct = sess.run(tf.one_hot(indices=y_train[0:2000],depth=10))

    for epoch in range(100):
        for i in range(100):
            input_x = numpy.matrix(data=x_train[i*20:(i+1)*20].reshape(-1,784))
            inpuy_y = numpy.matrix(data=correct[i*20:(i+1)*20])
            train_step.run(feed_dict={
                x:input_x,
                y_:inpuy_y
            })
        print(sess.run(fetches=correct_ration,
                       feed_dict={
                           x:numpy.matrix(data=x_train[:2000].reshape(-1,784)),
                           y_:numpy.matrix(data=correct)
                       }))

