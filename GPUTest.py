#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

#from keras import backend as K
#K.tensorflow_backend._get_available_gpus()

#import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 为使用CPU
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#import tensorflow as tf
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
