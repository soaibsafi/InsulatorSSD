"""
Use:
python train_test_split.py -i [PATH_TO_IMAGES_FOLDER]
"""
 
import os
import shutil
import numpy as np
import argparse


def train_test_split(path):
    ROOT_DIR = path
    os.makedirs(ROOT_DIR +'train/')
    os.makedirs(ROOT_DIR +'validation/')
    os.makedirs(ROOT_DIR +'test/')

    # Creating partitions of the data after shuffeling
    data_src = ROOT_DIR + 'Insulator/' # Folder to copy images from
    rootFilePath = []
    fn = []
    for name in data_src:
        fn=[os.path.splitext(filename)[0] for filename in os.listdir(data_src)]
    fn = list(set(fn))
    for f in fn:
        rootFilePath.append(data_src+f)
    print('Total images: ',len(rootFilePath))

    val_ratio = 0.20
    test_ratio = 0.10
    np.random.shuffle(rootFilePath)
    train_FileNames, val_FileNames, test_FileNames = np.split(np.array(rootFilePath),
                                                          [int(len(rootFilePath)* (1 - (val_ratio + test_ratio))), 
                                                           int(len(rootFilePath)* (1 - test_ratio))])
    
    print('Total images: ', len(rootFilePath))
    print('Training: ', len(train_FileNames))
    print('Validation: ', len(val_FileNames))
    print('Testing: ', len(test_FileNames))


    # Copy-pasting images
    for name in train_FileNames:
        shutil.copy(name + '.jpg', ROOT_DIR +'train/')
        shutil.copy(name + '.xml', ROOT_DIR +'train/')

    for name in val_FileNames:
        shutil.copy(name + '.jpg', ROOT_DIR +'validation/')
        shutil.copy(name + '.xml', ROOT_DIR +'validation/')

    for name in test_FileNames:
        shutil.copy(name + '.jpg', ROOT_DIR +'test/')
        shutil.copy(name + '.xml', ROOT_DIR +'test/')

def main():
    # Initiate argument parser
    parser = argparse.ArgumentParser(
        description="Split the dataset into train, test and validation directory"
    )

    parser.add_argument(
        "-i",
        "--inputDir",
        help="Path to the folder where the input images are stored",
        type=str
    )

    args = parser.parse_args()


    assert os.path.isdir(args.inputDir)

    train_test_split(args.inputDir)

    print("Successfully splited the dataset.")


if __name__ == "__main__":
    main()