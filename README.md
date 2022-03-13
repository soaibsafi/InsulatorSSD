# InsulatorSSD

## Split the dataset
```
!python /content/InsulatorSSD/helpers/active_learning.py -i /content/data/ -I
```

## Create csv from the xml annotation
```
!python /content/InsulatorSSD/helpers/xml_to_csv.py -i /content/data/train -o /content/data/record/train_labels.csv -l /content/data/record/

!python /content/InsulatorSSD/helpers/xml_to_csv.py -i /content/data/validation -o /content/data/record/validation_labels.csv -l /content/data/record/

!python /content/InsulatorSSD/helpers/xml_to_csv.py -i /content/data/test -o /content/data/record/test_labels.csv -l /content/data/record/
```

## Gererate tfrecord
```
!python /content/InsulatorSSD/helpers/generate_tfrecord.py /content/data/record/train_labels.csv /content/data/record/label_map.pbtxt /content/data/train /content/data/record/train.tfrecord

!python /content/InsulatorSSD/helpers/generate_tfrecord.py /content/data/record/validation_labels.csv /content/data/record/label_map.pbtxt /content/data/validation /content/data/record/validation.tfrecord

!python /content/InsulatorSSD/helpers/generate_tfrecord.py /content/data/record/test_labels.csv /content/data/record/label_map.pbtxt /content/data/test /content/data/record/test.tfrecord


```