# Insulator Detection Using a Single-Stage Detector (SSD)

## Training

### Build the Tensorflow API

1. Clone the tensorflow models on the colab- 

`!git clone --q https://github.com/tensorflow/models.git`

2. Navigate compile protos-

```
%cd models/research
!protoc object_detection/protos/*.proto --python_out=.
```

3. Install TensorFlow Object Detection API

```
!cp object_detection/packages/tf2/setup.py .
!python -m pip install .
```

4. Install COCO API

```
%cd /content
!pip install cython
!git clone https://github.com/cocodataset/cocoapi.git
%cd cocoapi/PythonAPI
!make
!cp -r pycocotools /content/models/research
```

5. Test the model Builder

```
%cd /content/models/research
!python object_detection/builders/model_builder_tf2_test.py
```

6. Download enx extract the pre-trained network
```
%cd /content
!wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz
!tar -xzvf ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz
```

### Data Processing

1. Split the dataset using Activte Learning Architecture
```
!python /content/InsulatorSSD/helpers/active_learning.py -i /content/data/ -I
```

2. Create csv from the xml annotation
```
!python /content/InsulatorSSD/helpers/xml_to_csv.py -i /content/data/train -o /content/data/record/train_labels.csv -l /content/data/record/

!python /content/InsulatorSSD/helpers/xml_to_csv.py -i /content/data/validation -o /content/data/record/validation_labels.csv -l /content/data/record/

!python /content/InsulatorSSD/helpers/xml_to_csv.py -i /content/data/test -o /content/data/record/test_labels.csv -l /content/data/record/
```

#### Gererate tfrecord
```
!python /content/InsulatorSSD/helpers/generate_tfrecord.py /content/data/record/train_labels.csv /content/data/record/label_map.pbtxt /content/data/train /content/data/record/train.tfrecord

!python /content/InsulatorSSD/helpers/generate_tfrecord.py /content/data/record/validation_labels.csv /content/data/record/label_map.pbtxt /content/data/validation /content/data/record/validation.tfrecord

!python /content/InsulatorSSD/helpers/generate_tfrecord.py /content/data/record/test_labels.csv /content/data/record/label_map.pbtxt /content/data/test /content/data/record/test.tfrecord

```

### Train the model

## Evaluation
[ ] ToDo
## Optimization
[ ] ToDo