"""
Rename the files
Usage:
 python .\helpers\rename_files.py -i [DIR_PATH]
"""

import os
import argparse


def rename(path):
    files = os.listdir(path)

    for index, file in enumerate(files):
        extension = os.path.splitext(file)[1]
        if extension=='.jpg':
            os.rename(os.path.join(path, file), os.path.join(path, 'Insulator_'+ str(index) + '.jpg'))
        elif extension =='.xml':
            os.rename(os.path.join(path, file), os.path.join(path, 'Insulator_'+ str(index) + '.xml'))


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

    rename(args.inputDir)

    print("Successfully renamed the dataset.")

if __name__ == "__main__":
    main()