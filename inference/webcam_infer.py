import os
import time
import numpy as np
import cv2
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

# Read label map -> TF2 changes
label_map_util.tf = tf.compat.v1
tf.gfile = tf.io.gfile

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging


PATH_TO_SAVED_MODEL = 'data/saved_model'
PATH_TO_LABELS = "data/record/label_map.pbtxt"
SCORE_THRESHOLD = 0.5

# Loading the model
print('Loading model...')
start_time = time.time()
# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
end_time = time.time()
elapsed_time = end_time - start_time
print('Model Loaded. Took {} seconds'.format(elapsed_time))

# Read the label
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


cap = cv2.VideoCapture(1)
while True:
    ret, image_np = cap.read()

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)


    # Take the first num_detections from the batch tensor
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}

    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    image_np_with_detections = image_np.copy()
    # visualize
    print(detections['detection_scores'])
    # https://github.com/tensorflow/models/blob/master/research/object_detection/utils/visualization_utils.py#L1101
    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=20,
          min_score_thresh=SCORE_THRESHOLD,
          agnostic_mode=False)

    cv2.imshow('Insulator Detection', cv2.resize(image_np_with_detections, (800, 600)))

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()