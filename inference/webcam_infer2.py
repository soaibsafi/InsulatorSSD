import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

# Read label map -> TF2 changes
label_map_util.tf = tf.compat.v1
tf.gfile = tf.io.gfile

def read_tensor_from_readed_frame(frame, input_height=300, input_width=300,
        input_mean=0, input_std=255):
  output_name = "normalized"
  float_caster = tf.cast(frame, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  #resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  resized = tf.image.resize(
    dims_expander,
    [input_height, input_width],
    method=tf.image.ResizeMethod.BILINEAR,
    preserve_aspect_ratio=False,
    antialias=False,
    name=None
)
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  #sess = tf.compat.v1.Session()
  #result = sess.run(normalized)
  return normalized

# def load_labels(label_file):
#   label = []
#   proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
#   for l in proto_as_ascii_lines:
#     label.append(l.rstrip())
#   return label

def VideoSrcInit(paath):
    cap = cv2.VideoCapture(paath)
    flag, image = cap.read()
    if flag:
        print("Valid Video Path. Lets move to detection!")
    else:
        raise ValueError("Video Initialization Failed. Please make sure video path is valid.")
    return cap

def main():
  PATH_TO_LABELS = "data/record/label_map.pbtxt"
  Model_Path = "data/model2.tflite"
  input_path = "video.mp4"

  ##Loading labels
  labels = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

  ##Load tflite model and allocate tensors
  interpreter = tf.lite.Interpreter(model_path=Model_Path)
  interpreter.allocate_tensors()
  # Get input and output tensors.
  input_details = interpreter.get_input_details()
  print(input_details)
  output_details = interpreter.get_output_details()

  input_shape = input_details[0]['shape']

  ##Read video
  cap = cv2.VideoCapture(0)

  while True:
    ok, image = cap.read()
    if not ok:
      break

    ##Converting the readed frame to RGB as opencv reads frame in BGR
    image = Image.fromarray(image).convert('RGB')
    tf.image.convert_image_dtype(image, dtype=np.uint8, saturate=False, name=None)

    ##Converting image into tensor
    image_tensor = read_tensor_from_readed_frame(image ,300, 300)
    #print(image_tensor)
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.uint8)


    ##Test model
    interpreter.set_tensor(input_details[0]['index'], image_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    ## You need to check the output of the output_data variable and 
    ## map it on the frame in order to draw the bounding boxes.


    cv2.namedWindow("cv_image", cv2.WINDOW_NORMAL)
    cv2.imshow("cv_image", image)

    ##Use p to pause the video and use q to termiate the program
    key = cv2.waitKey(10) & 0xFF
    if key == ord("q"):
      break
    elif key == ord("p"):
      cv2.waitKey(0)
      continue 
  cap.release()

if __name__ == '__main__':
  main()