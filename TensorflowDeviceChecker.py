from tensorflow.python.client import device_lib
from keras import backend as K

if __name__ == "__main__":
   print ("+++++++++++++++Tensorflow devices+++++++++++++++")
   print (device_lib.list_local_devices())
   print ("+++++++++++++++Keras GPU Check+++++++++++++++")
   print (K.tensorflow_backend._get_available_gpus())
