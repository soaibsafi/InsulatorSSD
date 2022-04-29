import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

def calculate_aspect_ratio(path):
    aspect_ratio = []
    for r, d, files in os.walk(path):
        for xml_file in glob.glob(r + "/*.xml"): 
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for member in root.findall('object'):
                bbox = member.findall('bndbox')[0]
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)

                dw = xmax - xmin
                dh = ymax - ymin

                if dw > dh:
                    height = dw
                    width = dh
                else:
                    height = dh
                    width = dw

                ratio = int(height/width)
                #print("{0}:{1}".format(first, second))
                aspect_ratio.append(ratio)
    return aspect_ratio
    
def plot_aspect_ratio(data):
    labels, counts =  np.unique(data, return_counts=True)

    x = np.arange(len(labels))
    width = .9  # the width of the bars
    fig = plt.figure(figsize = (10, 5))

    plt.bar(labels,counts, width, color='#66b3ff', label='Aspect Ratio')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel('No. of Objects')
    plt.xlabel('Aspect ratio')
    plt.title('Aspect Ratio of the training set')
    plt.xticks(x)
    plt.legend()
    plt.show()


def main():
    path = 'data/AugmentedImages/'
    aspect_ratio = calculate_aspect_ratio(path)
    plot_aspect_ratio(aspect_ratio)
    

if __name__ == "__main__":
    main()
