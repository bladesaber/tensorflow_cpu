import numpy
import datetime
import keras
from keras import layers

# 训练数据个数
training_examples = 10000
# 测试数据个数
testing_examples = 1000
# sin函数的采样间隔
sample_gap = 0.01
# 每个训练样本的长度
timesteps = 20
def generate_data(seq):
    '''
    生成数据，seq是一序列的连续的sin的值
    '''
    X = []
    y = []

    # 用前 timesteps 个sin值，估计第 timesteps+1 个
    # 因此， 输入 X 是一段序列，输出 y 是一个值
    for i in range(len(seq) - timesteps - 1):
        X.append(seq[i: i + timesteps])
        y.append(seq[i + timesteps])

    return numpy.array(X, dtype=numpy.float32), numpy.array(y, dtype=numpy.float32)

test_start = training_examples * sample_gap
test_end = test_start + testing_examples * sample_gap

# numpy.linspace 创建等差数列
train_x, train_y = generate_data(numpy.sin(numpy.linspace(0, test_start, training_examples)))
test_x, test_y = generate_data(numpy.sin(numpy.linspace(test_start, test_end, testing_examples)))

train_x = train_x.reshape((-1,20,1))
test_x = test_x.reshape((-1,20,1))

train_y = numpy.round(train_y,1)
test_y = numpy.round(test_y,1)
num_class = len(set(train_y))
train_y = keras.utils.to_categorical(train_y,num_class)
test_y = keras.utils.to_categorical(test_y,num_class)

model = keras.Sequential()
model.add(
    layers.GRU(units=100,use_bias=False,input_shape=(20,1),return_sequences=True,name='GRU_1')
)
model.add(layers.Dropout(0.3))
model.add(
    layers.GRU(100,use_bias=True,return_sequences=False,name='GRU_2')
)
model.add(layers.Dropout(0.3))
model.add(layers.Dense(units=num_class,use_bias=True,name='DENSE_1'))
model.add(layers.Activation('softmax'))

model.summary()

model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

model.fit(train_x, train_y,
          batch_size=256,
          epochs=5,
          verbose=1,
          shuffle=True,
          validation_data=[test_x,test_y])

#y = model.predict_on_batch(test_x)

#import matplotlib.pyplot as plt
#plt.plot(range(len(y)),y)
#plt.plot(range(len(y)),test_y)
#plt.show()