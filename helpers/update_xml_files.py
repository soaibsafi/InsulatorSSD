

import os
import glob
import argparse
import xml.etree.ElementTree as ET


def update_xml(path):
    for xml_file in glob.glob(path + "/*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        filename = root.find("filename")
        filename.text = os.path.basename(xml_file)

        filepath =  root.find("path")
        filepath.text = os.path.join(path, os.path.basename(xml_file))
            
        source = root.find("source")
        database = source.find("database")
        database.text = "APOLI"
        
        tree.write(xml_file,encoding='UTF-8')


def main():
    # Initiate argument parser
    parser = argparse.ArgumentParser(
        description="Sample TensorFlow XML-to-CSV converter"
    )
    parser.add_argument(
        "-i",
        "--inputDir",
        help="Path to the folder where the input .xml files are stored",
        type=str,
    )

    args = parser.parse_args()
    assert os.path.isdir(args.inputDir)
    update_xml(args.inputDir)



if __name__ == "__main__":
    main()
