import numpy
import matplotlib.pyplot as plt
from keras import backend
import keras
from keras.models import Sequential
from keras.layers import Conv2D,Dense,Dropout,MaxPooling2D,Flatten
from keras.optimizers import RMSprop,Adadelta
from keras.initializers import RandomNormal
from keras.regularizers import l2
from keras.losses import categorical_crossentropy

'''
Trains a simple convnet on the MNIST dataset.
Gets to 98% test accuracy after 10 epochs
(there is still a lot of margin for parameter tuning).
20 seconds per epoch on a CPU Core i5 7500.
3 seconds per epoch on a GPU 
'''

#-------------------------------------------- Step one Download DataSet
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

image_row = 28
image_col = 28
if backend.image_data_format()=='channels_first':
    x_train = x_train.reshape(x_train.shape[0],1,image_row,image_col)
    x_test = x_test.reshape(x_test.shape[0],1,image_row,image_col)
    input_shape = (1,image_row,image_col)
else:
    x_train = x_train.reshape(x_train.shape[0],image_row,image_col,1)
    x_test = x_test.reshape(x_test.shape[0],image_row,image_col,1)
    input_shape = (image_row,image_col,1)

#----------------------------------------- Step two Preprocessing
# 归一化，非常重要
x_train /= 255
x_test /= 255

#---------------------------------------- Step Three Create Dictory
num_class = len(set(y_train))
y_train = keras.utils.to_categorical(y=y_train,num_classes=num_class)
y_test = keras.utils.to_categorical(y=y_test,num_classes=num_class)

#--------------------------------------- Step foue Create Model Network
model = Sequential()
model.add(Conv2D(filters=20,kernel_size=(3,3),
                 activation='relu',
                 input_shape=input_shape,
                 padding='same',use_bias=False,
                 kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None),
                 bias_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None),
                 kernel_regularizer=l2(l=0.02)
                 ))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=10,kernel_size=(3,3),
                 activation='relu',
                 use_bias=False,
                 padding='valid',
                 kernel_regularizer=l2(0.01)))
model.add(MaxPooling2D(3,3))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(units=108,use_bias=True,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=num_class,use_bias=False,activation='softmax'))

#----------------------------------- Step five Choose Optimizer and Stop Rule and Loss function
model.compile(loss=categorical_crossentropy,
              optimizer=Adadelta(),
              #optimizer=RMSprop(),
              metrics=['accuracy'])

#----------------------------------- Step six Choose batch_size and Train times
model.fit(x=x_train,y=y_train,
          batch_size=150,
          epochs=3,
          verbose=1)

#----------------------------------- Evaluate model
score = model.evaluate(x_test,y_test,verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])