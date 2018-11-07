import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

max_steps = 1000
learning_rate = 0.001
dropout = 0.9

data_dir = 'C:/Users/bladesaber/Desktop/tensorflow_cpu/TensorflowExam/'
log_dir = 'C:/Users/bladesaber/Desktop/tensorflow_cpu/TensorflowExam/'

mnist = input_data.read_data_sets(train_dir=data_dir,one_hot=True)
sess = tf.InteractiveSession()

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(value=0.1,shape=shape)
    return tf.Variable(initial)

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        # summary 适用于 tensorbroad 这里是整理，总结
        tf.summary.scalar('mean',mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev',stddev)
        tf.summary.scalar('max',tf.reduce_max(var))
        tf.summary.scalar('min',tf.reduce_min(var))
        # 这里使用直方图
        tf.summary.histogram('histogram',var)

def nn_layer(input_tensor,input_dim,output_dim,layer_name,action=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable(shape=[input_dim,output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable(shape=[output_dim])
            variable_summaries(biases)
        with tf.name_scope('wx_plus_b'):
            pre_activate = tf.matmul(input_tensor,weights) + biases
            tf.summary.histogram('pre_activations',pre_activate)
        activations = action(pre_activate,name='activation')
        tf.summary.histogram('activation',activations)
        return activations

input_x = tf.placeholder(dtype=tf.float32,shape=[None,784])
input_y = tf.placeholder(dtype=tf.int32,shape=[None,1])

hidden1 = nn_layer(input_x,784,500,'layer1')
with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    dropped = tf.nn.dropout(hidden1,keep_prob)
y = nn_layer(dropped,500,10,'layer2',action=tf.identity)

with tf.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=input_y)
    with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(diff)
tf.summary.scalar('cross_entropy',cross_entropy)

with tf.name_scope('train'):
    # tf.train.Optimizer 竟然返回 train_step
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.arg_max(y,1),tf.arg_max(input_y,1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
tf.summary.scalar('accuracy',accuracy)

# 这里整合所有的 summary.scalar 操作
merged = tf.summary.merge_all()
# 将Session的计算图，sess.graph 加入记录器，这样在 GRAPHS 窗口可以看到计算图的可视化效果
train_writer = tf.summary.FileWriter(logdir=log_dir+'train',graph=sess.graph)
test_writer = tf.summary.FileWriter(logdir=log_dir+'test')

init = tf.global_variables_initializer()
sess.run(init)

def feed_dict(train):
    if train:
        xs,ys = mnist.train.next_batch(100)
        k = dropout
    else:
        xs,ys = mnist.test.images,mnist.test.labels
        k = 1.0
    return {
        input_x:xs,
        input_y:ys,
        keep_prob:k
    }

saver = tf.train.Saver()
for i in range(max_steps):
    if i % 10 == 0:
        summary,accuracy = sess.run(fetches=[merged,accuracy],
                                    feed_dict=feed_dict(False))
        test_writer.add_summary(summary,i)
        print('Accuracy at step %s : %s' % (i,accuracy))
    else:
        if i % 100 == 99:
            # FULL_TRACE 记录运算时间与内存占用
            run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
            # RunMetadata 用于记录 元信息
            run_metadata = tf.RunMetadata()
            summary,_ = sess.run(fetches=[merged,train_step],
                                 feed_dict=feed_dict(True),
                                 options=run_options,
                                 run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata=run_metadata,
                                          tag='step%03d' % i)
            train_writer.add_summary(summary,i)
            saver.save(sess=sess,save_path=log_dir+'model.ckpt',global_step=i)
            print('Adding run metadata for ',i)
        else:
            summary,_ = sess.run(fetches=[merged,train_step],
                                 feed_dict=feed_dict(True))
            train_writer.add_summary(summary=summary,global_step=i)
train_writer.close()
test_writer.close()

# Starting Tensorbroad in Broswer







