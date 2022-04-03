import tensorflow as tf

PATH_TO_SAVED_MODEL = 'data/saved_model'
PATH_TO_TFLITE_MODEL = 'data/'

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(PATH_TO_SAVED_MODEL) # path to the SavedModel directory
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter.allow_custom_ops=True
tflite_model = converter.convert()

# Save the model.
with open(PATH_TO_TFLITE_MODEL+'model2.tflite', 'wb') as f:
  f.write(tflite_model)