
import os
from keras.datasets import cifar10
import keras
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D
from keras.optimizers import Adadelta,RMSprop
from keras.losses import categorical_crossentropy

'''
Trains a simple convnet on the MNIST dataset.
Gets to 59.3% test accuracy after 2 epochs
(there is still a lot of margin for parameter tuning).
100 seconds per epoch on a CPU Core i5 7500.
'''

def paint(Metrix):
    plt.imshow(Metrix)
    plt.show()

save_dir = 'C:/Users/bladesaber\Desktop\TensorflowTest\Save_Model/'
model_name = 'keras_cifar10_trained_model.h5'

#--------------------------    Step  One  Download Dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data_from_LocakDisk()
input_shape = x_train.shape[1:]

#------------------------   Step Three  Create Dictory
num_class = len(set(y_train.reshape(1,-1)[0]))
y_train = keras.utils.to_categorical(y_train,num_class)
y_test = keras.utils.to_categorical(y_test,num_class)

#------------------------   Step Four   Create Model
model = Sequential()
model.add(Conv2D(filters=32,
                 kernel_size=(4,4),
                 padding='same',
                 input_shape=input_shape,
                 activation='relu'))
#model.add(Conv2D(filters=32,kernel_size=(4,4),padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=32,
                 kernel_size=(3,3),
                 padding='valid',
                 activation='relu'))
#model.add(Conv2D(filters=32,kernel_size=(3,3),padding='valid'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(units=800,
                activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=200,activation='relu',use_bias=True))
model.add(Dense(units=num_class,
                activation='softmax'))

model.summary()

#------------------------   Step Five   Choose Optimister and Stop Rule and Loss Function
model.compile(loss=categorical_crossentropy,
              optimizer=Adadelta(),
              metrics=['accuracy'])

#------------------------   Step Six    Choose Batch Size and Train epochs
# 数据增广
data_augmentation = False
if not data_augmentation:
    print('NO data augumentation')

    # ------------------------    Step Two   Propressing
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train / 255
    x_test = x_test / 255

    model.fit(x=x_train,y=y_train,
              batch_size=15,
              verbose=1,
              epochs=2,
              shuffle=True)
else:
    print('Use data augumentation')

    # ------------------------    Step Two   Propressing
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train / 255
    x_test = x_test / 255
    data_generator = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False
    )
    data_generator.fit(x_train)

    model.fit_generator(data_generator.flow(x=x_train,y=y_train,
                                            batch_size=15),
                        verbose=1,
                        epochs=2,
                        workers=4)

#------------------------   Evaluate
model_path = save_dir + model_name
model.save(model_path)

scores = model.evaluate(x=x_test,y=y_test,verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# -----------------    评价
# Batch Size 如果太小，相当于强化了每个样本对梯度的影响，随机方向的梯度优化可能存在并降低效率，并不好