import os
import fnmatch
import numpy as np
import matplotlib.pyplot as plt


GOOD_CONF_DIR = "/content/data/AugmentedImages/Good/"
AVERAGE_CONF_DIR = "/content/data/AugmentedImages/Average/"
BAD_CONF_DIR = "/content/data/AugmentedImages/Bad/"
LAB_CONF_DIR = "/content/data/AugmentedImages/Lab/"


def get_total_images(path):
    count = 0
    for r, d, files in os.walk(path):
        count += len(fnmatch.filter(os.listdir(r), '*.jpg'))
    return count


def main():
    labels = ['Good', 'Average','Bad','Lab']
    images = np.array([
        get_total_images(GOOD_CONF_DIR), 
        get_total_images(AVERAGE_CONF_DIR),
        get_total_images(BAD_CONF_DIR),
        get_total_images(LAB_CONF_DIR)
        ])
    print(images)
    colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']

    def absolute_value(val):
        a  = np.round(val/100*images.sum(), 0)
        return '{:.1f}%\n({:.0f})'.format(val, a)

    fig1, ax1 = plt.subplots()
    ax1.pie(images, colors = colors, autopct=absolute_value, startangle=90, pctdistance=.8)
    #patches, texts = ax1.pie(images, startangle=90)
    plt.legend(labels,  loc="best")
    ax1.axis('equal')  
    ax1.set_title("Dataset Visualization")
    
    plt.tight_layout()
    plt.show()
    


if __name__ == "__main__":
    main()