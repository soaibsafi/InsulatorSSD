import time
import pandas as pd
import re
import tensorflow as tf
from object_detection.utils import label_map_util
label_map_util.tf = tf.compat.v1
tf.gfile = tf.io.gfile
from object_detection.utils import visualization_utils as viz_utils


PATH_TO_SAVED_MODEL = "data/saved_model"
PATH_TO_LABELS = "data/record/label_map.pbtxt"
SCORE = 0.5

# Loading the model
print('Loading model...', end='')
start_time = time.time()
# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# Loading category index
#category_index = label_map_util.load_labelmap(PATH_TO_LABELS)
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)
print(category_index)


import glob
import os
files_grabbed = glob.glob(os.path.join('data/test', '*.jpg'))
IMAGE_PATHS = files_grabbed

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore') 
import cv2  

test_pred = pd.DataFrame()


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))


for image_path in IMAGE_PATHS:
    image_np = load_image_into_numpy_array(image_path)

    height, width, color = image_np.shape

    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    #print(detections['detection_scores'])

    for i in range(num_detections): 
      ll = []
      ll.append((image_path.split('\\'))[-1])
      ll.append(width)
      ll.append(height)
      ll.append(detections['detection_boxes'][i].tolist())
      ll.append("Insulator")
      ll.append(detections['detection_scores'][i])
      pre_df =pd.DataFrame([{'filename':ll[0],'width':ll[1],'height':ll[2],'class':ll[4],'xmin':ll[3][1] * width,'ymin':ll[3][0] * height,'xmax':ll[3][3] * width,'ymax':ll[3][2] * height,'score':ll[5]}])
      test_pred = pd.concat([test_pred,pre_df],axis=0)

    test_pred.to_csv("data/record/test_pred.csv",index=False)
    print('Detection record is crested at data/record/test_pred.csv')