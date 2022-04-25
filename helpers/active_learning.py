"""
Use:
python .\helpers\train_test_split.py -i [PATH_TO_IMAGES_FOLDER]
"""
 
import os
import time
import glob
import shutil
import numpy as np
import argparse
import fnmatch
import random

GOOD_CONF_DIR = "/content/data/AugmentedImages/Good/"
AVERAGE_CONF_DIR = "/content/data/AugmentedImages/Average/"
BAD_CONF_DIR = "/content/data/AugmentedImages/Bad/"
LAB_CONF_DIR = "/content/data/AugmentedImages/Lab/"
ROOT_DIR = "/content/data/"
INITIAL_SPLIT = True
ITERATION = 0


def get_total_images(path):
    count = 0
    for r, d, files in os.walk(path):
        count += len(fnmatch.filter(os.listdir(r), '*.jpg'))
    return count

def split_ratio(selected_for_training):
    previous_files = read_train_test()

    print("-------------------------")
    files = []
    fn=[os.path.splitext(filename)[0] for filename in os.listdir(GOOD_CONF_DIR)]
    fn = list(set(fn))
    for f in fn:
        files.append(GOOD_CONF_DIR+f)
    files = set(files).difference(previous_files)
    good_conf_ratio = int(selected_for_training * 0.45)
    good_conf_files = random.sample(files, good_conf_ratio)
    print("Good Confidence: ", len(good_conf_files))


    files = []
    fn=[os.path.splitext(filename)[0] for filename in os.listdir(AVERAGE_CONF_DIR)]
    fn = list(set(fn))
    for f in fn:
        files.append(AVERAGE_CONF_DIR+f)
    files = set(files).difference(previous_files)
    avg_conf_ratio = int(selected_for_training * 0.30)
    avg_conf_files = random.sample(files, avg_conf_ratio)
    print("Average Confidence: ", len(avg_conf_files))


    files = []
    fn=[os.path.splitext(filename)[0] for filename in os.listdir(BAD_CONF_DIR)]
    fn = list(set(fn))
    for f in fn:
        files.append(BAD_CONF_DIR+f)
    files = set(files).difference(previous_files)
    bad_conf_ratio = int(selected_for_training * 0.15)
    bad_conf_files = random.sample(files, bad_conf_ratio)
    print("Bad Confidence: ", len(bad_conf_files))

    files = []
    fn=[os.path.splitext(filename)[0] for filename in os.listdir(LAB_CONF_DIR)]
    fn = list(set(fn))
    for f in fn:
        files.append(LAB_CONF_DIR+f)
    files = set(files).difference(previous_files)
    lab_conf_ratio = int(selected_for_training * 0.10)
    lab_conf_files = random.sample(files, lab_conf_ratio)
    print("Lab Confidence: ", len(lab_conf_files))

    return good_conf_files+avg_conf_files+bad_conf_files+lab_conf_files

def write_selected_files(train_fileames, val_fileames, test_fileames):
    with open(ROOT_DIR +'train_iteration-'+str(ITERATION)+'.txt', 'w') as f:
        for name in train_fileames:
            f.write("%s\n" % name)
    with open(ROOT_DIR +'validation_iteration-'+str(ITERATION)+'.txt', 'w') as f:
        for name in val_fileames:
            f.write("%s\n" % name)
    with open(ROOT_DIR +'test_iteration-'+str(ITERATION)+'.txt', 'w') as f:
        for name in test_fileames:
            f.write("%s\n" % name)        

def read_train_test():
    trainfile_list = glob.glob(os.path.join(ROOT_DIR, 'train_*.txt'))
    trainfiles = []
    for file in trainfile_list:
        with open(file, 'r') as f:
            trainfiles += f.read().splitlines()
    
    validationfile_list = glob.glob(os.path.join(ROOT_DIR, 'validation_*.txt'))
    validationfiles = []
    for file in validationfile_list:
        with open(file, 'r') as f:
            validationfiles += f.read().splitlines()
    
    testfile_list = glob.glob(os.path.join(ROOT_DIR, 'test_*.txt'))
    testfiles = []
    for file in testfile_list:
        with open(file, 'r') as f:
            testfiles += f.read().splitlines()

    return trainfiles+validationfiles+testfiles

def copy_selected_data(train_fileames, val_fileames, test_fileames):
    for item in train_fileames:
            shutil.move(item + '.jpg', ROOT_DIR +'train/')
            shutil.move(item + '.xml', ROOT_DIR +'train/')
    for item in val_fileames:
            shutil.move(item + '.jpg', ROOT_DIR +'validation/')
            shutil.move(item + '.xml', ROOT_DIR +'validation/')
    for item in test_fileames:
            shutil.move(item + '.jpg', ROOT_DIR +'test/')
            shutil.move(item + '.xml', ROOT_DIR +'test/')

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

    write_selected_files(train_FileNames, val_FileNames, test_FileNames)
    return train_FileNames, val_FileNames, test_FileNames


def main():
    global INITIAL_SPLIT
    global ITERATION
    # Initiate argument parser
    parser = argparse.ArgumentParser(
        description="Split the dataset into train, test and validation directory"
    )
    parser.add_argument(
        "-dir",
        "--inputDir",
        help="Path to the folder where the input images are stored",
        type=str
    )
    parser.add_argument(
        "-I",
        "--iteration",
        help="Path to the folder where the input images are stored",
        type=int
    )
    args = parser.parse_args()
    assert os.path.isdir(args.inputDir)

    print("-------------------------")
    total_images = get_total_images(args.inputDir)
    print("Total Images: ",total_images)

    ITERATION = args.iteration

    if args.iteration==1:
        INITIAL_SPLIT = True
        selection_ratio = 0.15
    else:
        INITIAL_SPLIT = False
        selection_ratio = 0.04
    

    selected_for_training = int(total_images * selection_ratio)
    print("Selected for Training: ",selected_for_training)

    selected_files = split_ratio(selected_for_training)
    train_fileames, val_fileames, test_fileames = train_test_split(selected_files)
    copy_selected_data(train_fileames, val_fileames, test_fileames)

    print("-------------------------")
    print("Split Successful. Current details-")
    print('Training: ', get_total_images(ROOT_DIR +'train/'))
    print('Validation: ', get_total_images(ROOT_DIR +'validation/'))
    print('Testing: ', get_total_images(ROOT_DIR +'test/'))
    print("-------------------------")

if __name__ == "__main__":
    main()
