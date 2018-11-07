
import keras
from keras.layers import Dense,Conv2D,BatchNormalization,Activation,Input,MaxPooling2D,Flatten,AveragePooling2D
from keras.models import Model
from keras.losses import categorical_crossentropy
from keras.datasets import cifar10
from keras.callbacks import ModelCheckpoint,LearningRateScheduler,ReduceLROnPlateau,TensorBoard
from keras.regularizers import l2
from keras.optimizers import Adam

#------------------------------- Step one Download DataSet
(x_train, y_train), (x_test, y_test) = cifar10.load_data_from_LocakDisk()

#------------------------------- Step two Preprocessing
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train/255
x_test = x_test/255
input_shape = x_train.shape[1:]

#------------------------------- Step three Create Dictory
num_class = 10
y_train = keras.utils.to_categorical(y=y_train,num_classes=num_class)
y_test = keras.utils.to_categorical(y=y_test,num_classes=num_class)

#------------------------------- Step four Create Model
def resnet_layer(inputs,
                 num_filter = 16,
                 kernel_size = 3,
                 strides = 1,
                 activation = 'relu',
                 batch_normalization = True,
                 conv_first = True):
    conv = Conv2D(filters=num_filter,
                  kernel_size=kernel_size,
                  strides = strides,
                  padding='same',
                  use_bias=False,
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))
    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation != None:
            x = Activation(activation=activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation != None:
            x = Activation(activation=activation)(x)
    return x

def origin_resnet_block(inputs,num_filter,input_resize):

    y = resnet_layer(inputs=inputs,
                 num_filter=num_filter,
                 kernel_size=2,
                 batch_normalization=True,
                 conv_first=True)
    y = resnet_layer(inputs=y,
                     num_filter=num_filter,
                     kernel_size=2,
                     batch_normalization=True,
                     conv_first=True)
    y = resnet_layer(inputs=y,
                     num_filter = num_filter,
                     kernel_size=2,
                     batch_normalization=True,
                     conv_first=True,
                     activation=None)

    if input_resize:
        x = resnet_layer(inputs=inputs,
                         num_filter=num_filter,
                         kernel_size=1,
                         activation=None,
                         batch_normalization=False)
        y = keras.layers.add([x,y])
    else:
        y = keras.layers.add([inputs, y])
    y = Activation('relu')(y)
    return y

def resnet():
    inputs = Input(shape=input_shape)
    y = Conv2D(filters=64,
           kernel_size=3,
           strides=1,
           padding='same',use_bias=False,
           kernel_initializer='he_normal',
           kernel_regularizer=l2(1e-4))(inputs)
    y = MaxPooling2D(pool_size=2)(y)

    y = origin_resnet_block(inputs=y,num_filter=16,input_resize=True)
    y = origin_resnet_block(inputs=y,num_filter=16,input_resize=False)
    y = origin_resnet_block(inputs=y,num_filter=16,input_resize=False)
    
    y = MaxPooling2D(pool_size=2)(y)
    y = origin_resnet_block(inputs=y,num_filter=32,input_resize=True)
    y = origin_resnet_block(inputs=y,num_filter=32,input_resize=False)
    y = origin_resnet_block(inputs=y,num_filter=32,input_resize=False)

    y = MaxPooling2D(pool_size=2)(y)
    y = origin_resnet_block(inputs=y,num_filter=64,input_resize=True)
    y = origin_resnet_block(inputs=y,num_filter=64,input_resize=False)
    y = origin_resnet_block(inputs=y,num_filter=64,input_resize=False)

    y = AveragePooling2D(pool_size=2)(y)
    y = Flatten()(y)

    output = Dense(units=num_class,
              use_bias=True,
              activation='softmax',
              kernel_initializer='he_normal')(y)

    model = Model(inputs=inputs,outputs=output)
    return model

def resnet_block(inputs,num_filter,input_resize):

    y = resnet_layer(inputs=inputs,
                     num_filter=num_filter,
                     kernel_size=3,
                     batch_normalization=True,
                     conv_first=True)
    y = resnet_layer(inputs=y,
                     num_filter=num_filter,
                     kernel_size=3,
                     batch_normalization=True,
                     conv_first=True,
                     activation=None)

    if input_resize:
        x = resnet_layer(inputs=inputs,
                         num_filter=num_filter,
                         kernel_size=1,
                         activation=None,
                         batch_normalization=False)
        y = keras.layers.add([x, y])
    else:
        y = keras.layers.add([inputs, y])
    y = Activation('relu')(y)
    return y

def resnet_v1_keras():
    inputs = Input(shape=input_shape)

    y = resnet_layer(inputs=inputs,num_filter=16)

    y = resnet_block(inputs=y, num_filter=16,input_resize=False)
    y = resnet_block(inputs=y, num_filter=16,input_resize=False)
    y = resnet_block(inputs=y, num_filter=16, input_resize=False)

    y = resnet_block(inputs=y, num_filter=32, input_resize=True)
    y = resnet_block(inputs=y, num_filter=32, input_resize=False)
    y = resnet_block(inputs=y, num_filter=32, input_resize=False)

    y = resnet_block(inputs=y, num_filter=64, input_resize=True)
    y = resnet_block(inputs=y, num_filter=64, input_resize=False)
    y = resnet_block(inputs=y, num_filter=64, input_resize=False)

    y = AveragePooling2D(pool_size=8)(y)
    y = Flatten()(y)
    outputs = Dense(num_class,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    model = Model(inputs=inputs,outputs=outputs)
    return model

model = resnet()
#model = resnet_v1_keras()
model.summary()

#------------------------------- Step five Choose stop rule and optimizer
model.compile(loss=categorical_crossentropy,
              optimizer=Adam(),
              metrics=['accuracy'])

save_dir = 'C:/Users/bladesaber\Desktop/tensorflow_cpu/Save_Model'
model_name = 'keras_cifar10_resnet_v1_model.h5'
filepath = save_dir+model_name

def lr_schedule(epoch):
    lr = 1e-2
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)
lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=5,
                               mode='auto',
                               cooldown=0,
                               min_lr=0.5e-6)
lr_scheduler = LearningRateScheduler(lr_schedule)

Log_dir = 'C:/Users/bladesaber/Desktop/tensorflow_cpu/Log/'
file_name = 'cifar10_resnet_log'
tb = TensorBoard(log_dir= Log_dir+file_name,write_grads=True)

callbacks = [checkpoint,lr_reducer,lr_scheduler,tb]

#------------------------------- Step six Choose batch Size and epoches
model.fit(x=x_train,y=y_train,
          batch_size=32,
          epochs=2,
          shuffle=True,
          callbacks=callbacks)

#------------------------------- Evaluate
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])