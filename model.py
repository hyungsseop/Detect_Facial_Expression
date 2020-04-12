import tensorflow as tf
from tensorflow import keras

new_model = keras.models.load_model('Trained_Model.h5')
new_model.summary()


from keras.utils import plot_model
plot_model(new_model, to_file='/home/hyungseop/test/model.png')