"""
Use:
python .\helpers\train_test_split.py -i [PATH_TO_IMAGES_FOLDER]
"""
 
import os
import shutil
import numpy as np
import argparse
import fnmatch
import random


GOOD_CONF_DIR = "/content/data/Good/"
AVERAGE_CONF_DIR = "/content/data/Average/"
BAD_CONF_DIR = "/content/data/Bad/"
LAB_CONF_DIR = "/content/data//Lab/"
#TRAINING_DATA_DIR = "data/Insulator/"
ROOT_DIR = "/content/data/"


def get_total_images(path):
    count = 0
    for r, d, files in os.walk(path):
        count += len(fnmatch.filter(os.listdir(r), '*.jpg'))
    return count

def split_ratio(selected_for_training):
    print("-------------------------")
    files = []
    fn=[os.path.splitext(filename)[0] for filename in os.listdir(GOOD_CONF_DIR)]
    fn = list(set(fn))
    for f in fn:
        files.append(GOOD_CONF_DIR+f)
    good_conf_ratio = int(selected_for_training * 0.45)
    good_conf_files = random.sample(files, good_conf_ratio)
    print("Good Confidence: ", len(good_conf_files))


    files = []
    fn=[os.path.splitext(filename)[0] for filename in os.listdir(AVERAGE_CONF_DIR)]
    fn = list(set(fn))
    for f in fn:
        files.append(AVERAGE_CONF_DIR+f)
    avg_conf_ratio = int(selected_for_training * 0.30)
    avg_conf_files = random.sample(files, avg_conf_ratio)
    print("Average Confidence: ", len(avg_conf_files))


    files = []
    fn=[os.path.splitext(filename)[0] for filename in os.listdir(BAD_CONF_DIR)]
    fn = list(set(fn))
    for f in fn:
        files.append(BAD_CONF_DIR+f)
    bad_conf_ratio = int(selected_for_training * 0.15)
    bad_conf_files = random.sample(files, bad_conf_ratio)
    print("Bad Confidence: ", len(bad_conf_files))

    files = []
    fn=[os.path.splitext(filename)[0] for filename in os.listdir(LAB_CONF_DIR)]
    fn = list(set(fn))
    for f in fn:
        files.append(LAB_CONF_DIR+f)
    lab_conf_ratio = int(selected_for_training * 0.10)
    lab_conf_files = random.sample(files, lab_conf_ratio)
    print("Lab Confidence: ", len(lab_conf_files))

    return good_conf_files+avg_conf_files+bad_conf_files+lab_conf_files

def move_selected_data(selected_files):
    # for dir in selected_files:
    #     for item in dir:
    #         shutil.move(item + '.jpg', TRAINING_DATA_DIR)
    #         shutil.move(item + '.xml', TRAINING_DATA_DIR)

    with open('log\selected_files2.txt', 'w') as f:
        for dir in selected_files:
            for item in dir:
                f.write("%s\n" % item)

def train_test_split(selected_files):
    os.makedirs(ROOT_DIR +'train/', exist_ok=True)
    os.makedirs(ROOT_DIR +'validation/', exist_ok=True)
    os.makedirs(ROOT_DIR +'test/', exist_ok=True)


    # Creating partitions of the data after shuffeling
    val_ratio = 0.20
    test_ratio = 0.10
    np.random.shuffle(selected_files)
    train_FileNames, val_FileNames, test_FileNames = np.split(np.array(selected_files),
                                                          [int(len(selected_files)* (1 - (val_ratio + test_ratio))), 
                                                           int(len(selected_files)* (1 - test_ratio))])
    
    print("-------------------------")
    print('Training: ', len(train_FileNames))
    print('Validation: ', len(val_FileNames))
    print('Testing: ', len(test_FileNames))

    # with open('./log/train.txt', 'w') as f:
    #     for name in train_FileNames:
    #         f.write("%s\n" % name)

    # with open('./log/validation.txt', 'w') as f:
    #     for name in val_FileNames:
    #         f.write("%s\n" % name)
    # with open('./log/test.txt', 'w') as f:

    #     for name in test_FileNames:
    #         f.write("%s\n" % name)

    # Move the images
    for name in train_FileNames:
        shutil.move(name + '.jpg', ROOT_DIR +'train/')
        shutil.move(name + '.xml', ROOT_DIR +'train/')

    for name in val_FileNames:
        shutil.move(name + '.jpg', ROOT_DIR +'validation/')
        shutil.move(name + '.xml', ROOT_DIR +'validation/')

    for name in test_FileNames:
        shutil.move(name + '.jpg', ROOT_DIR +'test/')
        shutil.move(name + '.xml', ROOT_DIR +'test/')


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
    parser.add_argument(
        "-I",
        "--initialSplit",
        help="Path to the folder where the input images are stored",
        action='store_true'
    )
    args = parser.parse_args()
    assert os.path.isdir(args.inputDir)

    print("-------------------------")
    total_images = get_total_images(args.inputDir)
    print("Total Images: ",total_images)

    if args.initialSplit:
        selection_ratio = 0.15
    else:
        selection_ratio = 0.04

    selected_for_training = int(total_images * selection_ratio)
    print("Selected for Training: ",selected_for_training)

    selected_files = split_ratio(selected_for_training)
    train_test_split(selected_files)

    print("-------------------------")
    print("Split Successful. Current details-")
    print('Training: ', get_total_images(ROOT_DIR +'train/'))
    print('Validation: ', get_total_images(ROOT_DIR +'validation/'))
    print('Testing: ', get_total_images(ROOT_DIR +'test/'))
    print("-------------------------")
    print('Remaining Images: ', get_total_images('AugmentedImages/'))
    print("-------------------------")





if __name__ == "__main__":
    main()