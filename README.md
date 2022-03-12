# InsulatorSSD

## Split the dataset
```
python .\helpers\train_test_split.py -i .\data\
```

## Create csv from the xml annotation
```
python .\helpers\xml_to_csv.py -i .\data\train -o .\data\record\train_labels.csv -l .\data\record\

python .\helpers\xml_to_csv.py -i .\data\test -o .\data\record\test_labels.csv `
```

## Gererate tfrecord
```
python .\helpers\generate_tfrecord.py data\record\train_labels.csv data\record\label_map.pbtxt data\train data\record\train.tfrecord

python .\helpers\generate_tfrecord.py data\record\test_labels.csv data\record\label_map.pbtxt data\test data\record\test.tfrecord  
```