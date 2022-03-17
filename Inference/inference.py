import tensorflow as tf
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import matplotlib.pyplot as plt

SAVED_MODEL_PATH = 'TrainedModel\inference\saved_model'

detect_fn = tf.saved_model.load(SAVED_MODEL_PATH)
print(detect_fn)