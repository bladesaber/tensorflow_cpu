import sys
import tensorflow as tf
import numpy as np
import collections
import tensorflow.contrib.rnn as rnnTool
import tensorflow.contrib as contrib
import time

sess = tf.InteractiveSession()

class PTB_DataSet_Genrator(object):

    Py3 = sys.version_info[0] == 3

    def read_words(self,filename):
        with tf.gfile.GFile(filename,'r') as f:
            if self.Py3:
                return f.read().replace('\n','<eos>').split()
            else:
                return f.read().decode('utf-8').replace('\n','<eos>').split()

    def build_vocab(self,filename):
        data = self.read_words(filename=filename)

        # 生成一个频数统计的字典
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(),
                             key=lambda x:(-x[1],x[0]))

        # words 是 词列表
        words,_ = list(zip(*count_pairs))
        word_to_id = dict(zip(words,range(len(words))))

        #for i in count_pairs:
        #    print('build_vocab.count_pairs: ', i)
        #print('build_vocab.words: ', words)

        return word_to_id

    def file_to_word_ids(self,filename,word_to_id):
        data = self.read_words(filename)
        return [word_to_id[word] for word in data if word in word_to_id]

    def ptb_raw_data(self,data_path = None):
        """
        Load PTB raw data from data directory "data_path".
          Reads PTB text files, converts strings to integer ids,
          and performs mini-batching of the inputs.
          The PTB dataset comes from Tomas Mikolov's webpage:
          http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
          Args:
            data_path: string path to the directory where simple-examples.tgz has
              been extracted.
          Returns:
            tuple (train_data, valid_data, test_data, vocabulary)
            where each of the data objects can be passed to PTBIterator.
        """
        temportary_dir = 'C:/Users/bladesaber/Desktop/tensorflow_cpu/DataSet/PTB/'
        train_path = temportary_dir + 'ptb.train.txt'
        valid_path = temportary_dir + 'ptb.valid.txt'
        test_path = temportary_dir + 'ptb.test.txt'

        # word_to_id 为字典，有每一个word 及其对应 id
        word_to_id = self.build_vocab(train_path)
        # train_data 为 一个 word 转 id 后的列表
        train_data = self.file_to_word_ids(train_path,word_to_id)
        valid_data = self.file_to_word_ids(valid_path,word_to_id)
        test_data = self.file_to_word_ids(test_path,word_to_id)
        vocabulary = len(word_to_id)

        #for i in word_to_id:
        #    print('ptb_raw_data.word_to_id: ', i,': ',word_to_id[i])
        #print('ptb_raw_data.train_data: ', train_data)
        return train_data,valid_data,test_data,vocabulary

    def ptb_producer(self,raw_data,batch_size,num_steps,name=None):
        with tf.name_scope(name=name,
                           default_name='PTBProducer',
                           values=[raw_data,batch_size,num_steps]):
            # raw_data 是个单行 word_to_id 列表
            raw_data = tf.convert_to_tensor(
                value=raw_data,name='raw_data',dtype=tf.int32)
            data_len = tf.size(raw_data)

            sess.run(tf.Print(input_=data_len,data=[data_len],message='ptb_producer.data_len: '))

            batch_len = data_len // batch_size
            # 这里使用 batch_size * batch_len 是用于将 多余的小量数据去除
            # 这里的 batch_size 是指训练样本切分个数
            data = tf.reshape(tensor=raw_data[0:batch_size*batch_len],
                              shape=[batch_size,batch_len])
            # 这里的 epoch_size ？？
            epoch_size = (batch_len-1) // num_steps

            # 检查，断言为证 否则弹出
            assertion = tf.assert_positive(
                epoch_size,
                message="epoch_size == 0, decrease batch_size or num_steps")
            # tf.control_dependencies 是上下文管理器，指在执行 with 内语句前，必须先执行 [] 内内容 这里是 assertion
            # tf.identity 相当于 等于赋值 ，至于为什么不用 = ，日后再解
            with tf.control_dependencies([assertion]):
                epoch_size = tf.identity(epoch_size,
                                         name='epoch_size')

            # tf.train.range_input_producer 产生一个队列，当 shuffle=False 队列为顺序队列，大小为 0-limit ， num_epochs为出现次数，不指定时无限出现
            # 这个函数目前没法验证出来
            i = tf.train.range_input_producer(limit=epoch_size,shuffle=False).dequeue()
            # 这里 tf.strided_slice 相当于二维分割，begin = [x1,x2,x3...] , end = [y1,y2,y3...] 指第一维取 x1到y1 ，第二维取 x2到y2 以此类推
            # data 结构 [ batch_size,batch_len ]
            # i * num_steps ？？
            x = tf.strided_slice(input_=data,
                                 begin=[0,i*num_steps],
                                 end=[batch_size,(i+1)*num_steps])
            x.set_shape([batch_size,num_steps])
            y = tf.strided_slice(input_=data,
                                 begin=[0,i*num_steps+1],
                                 end=[batch_size,(i+1)*num_steps+1])
            y.set_shape([batch_size,num_steps])
            return x,y

class PTBInput(object):
    def __init__(self,config,data,name=None):
        self.batch_size = config.batch_size
        print('PTBInput.batch_size: ',config.batch_size)
        self.reader = PTB_DataSet_Genrator()

        # 指代 LSTM 展开步数
        self.num_steps = config.num_steps
        self.epoch_size = (len(data) // self.batch_size - 1 )//self.num_steps
        self.input_data,self.targets =self.reader.ptb_producer(raw_data=data,
                                                               batch_size=self.batch_size,
                                                               num_steps=self.num_steps,
                                                               name=name)

class PTBModel(object):
    def __init__(self,is_training,config,input_):
        self._input = input_
        batch_size = input_.batch_size
        num_steps = input_.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size

        def lstm_cell():
            # BasicLSTMCell
            # num_units 是指
            # state_is_tuple 官方建议设置为True。此时，输入和输出的states为c(cell状态)和h（输出）的二元组 , c 指长期记忆,h 指输出的状态
            # zero_state(batch_size, tf.float32) 指定batch_size,将c和h全部初始化为0，shape全是batch_size * num_units
            return rnnTool.BasicLSTMCell(
                num_units=size,
                forget_bias=0.0,
                state_is_tuple=True)

        attn_cell = lstm_cell
        if is_training and config.keep_prob < 1:
            def attn_cell():
                return rnnTool.DropoutWrapper(cell=lstm_cell(),
                                              output_keep_prob=config.keep_prob)
        cell = rnnTool.MultiRNNCell(
            cells=[attn_cell() for _ in range(config.num_layers)],
            state_is_tuple=True)
        self._initial_state = cell.zero_state(batch_size=batch_size,
                                              dtype=tf.float32)

        embedding = tf.get_variable(
            name='embedding',shape=[vocab_size,size],
            dtype=tf.float32)
        inputs = tf.nn.embedding_lookup(params=embedding,
                                        ids=input_.input_data)
        if is_training and config.keep_prob<1:
            inputs = tf.nn.dropout(x=inputs,keep_prob=config.keep_prob)

        outputs = []
        state = self._initial_state
        with tf.variable_scope('RNN'):
            for time_step in range(num_steps):
                if time_step > 0 :
                    # tf.get_variable_scope() 获取当前 scope，
                    # reuse_variables() 当前 scope 内的所有变量可复用
                    tf.get_variable_scope().reuse_variables()

                # state 这里是共用的，input 是按顺序注入数据
                # cell_output 是每次运算结果
                # 这里的写法很特别，直接使用（）相当于调用  __call__
                # 理论上这里的 cell_output 维度与 状态的维度相同(config.hidden_size)
                # 这里的 inputs 数据结构为： [batch_size , batch_len维度 , 词向量维度]
                (cell_output,state) = cell(inputs=inputs[:,time_step,:],
                                           state=state)
                outputs.append(cell_output)

        # 这里的 output 不知是什么操作
        output = tf.reshape(tf.concat(outputs,1),[-1,size])

        # 这里好奇怪， w 的 size 是 vocab_size 但 target 的 size 是 num_steps，先不做考虑
        softmax_w = tf.get_variable(name='softmax_w',
                                    shape=[size,vocab_size],
                                    dtype=tf.float32)
        softmax_b = tf.get_variable(name='softmax_b',
                                    shape=[vocab_size],
                                    dtype=tf.float32)
        logits = tf.matmul(output,softmax_w) + softmax_b
        loss = contrib.legacy_seq2seq.sequence_loss_by_example(
            logits=[logits],
            targets=[tf.reshape(tensor=input_.targets,shape=[-1])],
            weights=[tf.ones(shape=[batch_size * num_steps],dtype=tf.float32)])
        self._cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state

        if not is_training:
            return

        self._lr = tf.Variable(initial_value=0.0,trainable=False)
        tvars = tf.trainable_variables()
        grads,_ = tf.clip_by_global_norm(
            t_list=tf.gradients(ys=self._cost,xs=tvars),
            clip_norm=config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(zip(grads,tvars),
                                                   global_step=contrib.framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(dtype=tf.float32,shape=[],name='new_learning_rate')
        self._lr_update = tf.assign(self._lr,self._new_lr)

    def input(self):
        return self._input

    def initial_state(self):
        return self._initial_state

    def cost(self):
        return self._cost

    def final_state(self):
        return self._final_state

    def lr(self):
        return self._lr

    def train_op(self):
        return self._train_op

class SmallConfig(object):
    '''
    The hyperparameters used in the model:
    - init_scale - the initial scale of the weights
    - learning_rate - the initial value of the learning rate
    - max_grad_norm - the maximum permissible norm of the gradient
    - num_layers - the number of LSTM layers
    - num_steps - the number of unrolled steps of LSTM
    - hidden_size - the number of LSTM units
    - max_epoch - the number of epochs trained with the initial learning rate
    - max_max_epoch - the total number of epochs for training
    - keep_prob - the probability of keeping weights in the dropout layer
    - lr_decay - the decay of the learning rate for each epoch after "max_epoch"
    - batch_size - the batch size
    - rnn_mode - the low level implementation of lstm cell: one of CUDNN,
             BASIC, or BLOCK, representing cudnn_lstm, basic_lstm, and
             lstm_block_cell classes.
    '''
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000

def run_epoch(session,model,eval_op=None,verbose = False):

    # 这里跑 每一个 epoch 的运算

    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    fetch = {
        'cost':model.cost,
        'final_state':model.final_state
    }
    if eval_op is not None:
        fetch['eval_op'] = eval_op

    for step in range(model.input.epoch_size):
        feed_dict = {}
        for i,(c,h) in enumerate(model.initial_state):
            # 这里相当于 更新 长期记忆 与 状态列表
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetch,feed_dict)
        cost = vals['cost']
        state = vals['final_state']

        costs += cost
        iters += model.input.num_steps

        if verbose and step % (model.input.epoch_size // 10) == 10:
            print('%.3f perplexity: %.3f speed: %.0f wps' % (
                step*1.0/model.input.epoch_size,
                np.exp(costs/iters),
                iters * model.input.batch_size/(time.time()-start_time)
            ))

    return np.exp(costs / iters)

reader = PTB_DataSet_Genrator()
train_data,valid_data,test_data,vocabulary = reader.ptb_raw_data()
config = SmallConfig()

train_input = PTBInput(config=config,data=train_data,name='TrainInput')

with tf.Graph().as_default():
    initializer = tf.random_normal_initializer(mean=-config.init_scale,
                                               stddev=config.init_scale)
    with tf.name_scope('Train'):
        train_input = PTBInput(config=config,data=train_data,name='TrainInput')
        with tf.variable_scope(name_or_scope='Model',reuse=None,initializer=initializer):
            m = PTBModel(is_training=True,config=config,input_=train_input)

sv = tf.train.Supervisor()
with sv.managed_session() as sess:
    for i in range(config.max_max_epoch):
        lr_decay = config.lr_decay ** max(i+1-config.max_max_epoch,0.0)
        # 调控学习速率
        m.assign_lr(sess, config.learning_rate * lr_decay)

        print("Epoch: %d Learning rate: %.3f" % (i + 1, sess.run(m.lr)))
        train_perplexity = run_epoch(sess, m, eval_op=m.train_op,verbose=True)













