import tensorflow as tf
import keras
from keras.models import load_model

emotion_model = load_model('MonkeyPox_detection_model_50epochs.h5', compile=False)

converter = tf.lite.TFLiteConverter.from_keras_model(emotion_model)
#converter.optimizations = [tf.lite.Optimize.DEFAULT] #Uses default optimization strategy to reduce the model size
tflite_model = converter.convert()
open("Monkeypox_detection_model_50epochs_no_opt.tflite", "wb").write(tflite_model)