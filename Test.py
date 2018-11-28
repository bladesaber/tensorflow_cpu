from keras.models import Sequential
from keras import layers
import numpy

model = Sequential()
model.add(layers.TimeDistributed(layers.Dense(4), input_shape=(None, 2)))
model.summary()

data = numpy.reshape([1,2,1,2],(1,2,2))
result = model.predict(data)
print(result)