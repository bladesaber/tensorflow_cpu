#coding=utf-8

import keras
import csv
import numpy
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard

'''
Trains a simple deep NN on the MNIST dataset.
Gets to 98.53% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
4 seconds per epoch on a CPU Core i5 7500.
'''

#---------------------------------------------- Step one Download DataSet
class DataGenner(object):
    def dataSet_mini(self):
        path = 'C:/Users/bladesaber\Desktop\TensorflowTest\DataSet\mnist.npz'
        f = numpy.load(path)
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        f.close()
        return (x_train, y_train), (x_test, y_test)

class Paint(object):
    def plant(self,Metrix):
        plt.imshow(Metrix)
        plt.show()

(x_train, y_train), (x_test, y_test) = DataGenner().dataSet_mini()
x_train = x_train.reshape(len(x_train),784)
x_test = x_test.reshape(len(x_test),784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#---------------------------------------------- Step two Preprocessing
# 归一化，非常重要
x_train /= 255
x_test /= 255

#---------------------------------------------- Step three Create Dictory
num_of_class = len(set(y_train))
# one-hot 布局
y_train = keras.utils.to_categorical(y=y_train, num_classes=num_of_class)
y_test = keras.utils.to_categorical(y=y_test, num_classes=num_of_class)

#--------------------------------------------- Step four Create Network
model = Sequential()
# units 为输出维数
model.add(Dense(units=300,activation='relu',
                use_bias=True,
                input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(300,activation='relu',
                use_bias=True,))
model.add(Dropout(0.2))
model.add(Dense(num_of_class,activation='softmax'))

# 输出状态信息
model.summary()

#------------------------------------------ Step five Choose Optimizer and Stop_Rule
model.compile(loss='categorical_crossentropy',
              #optimizer=keras.optimizers.SGD(),
              optimizer = keras.optimizers.RMSprop(),
              metrics=['accuracy'])

Log_dir = 'C:/Users/bladesaber/Desktop/tensorflow_cpu/Log/'
file_name = 'minist_mlp_log'
tb = TensorBoard(log_dir=Log_dir+file_name,write_grads=True)

#------------------------------------------ Step six Choose Batch_Size and Train_Time
batch_size = 150
history = model.fit(x=x_train,y=y_train,
                    batch_size=batch_size,
                    epochs=10,
                    verbose=1,
                    shuffle=True,
                    validation_data=(x_test,y_test),
                    callbacks=[tb])

#----------------------------------------- Step seven Evalute the Model
score = model.evaluate(x=x_test,y=y_test,verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])