import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

ROOT_DIR = "/content/data/"
ITERATION = 0


def prepare_dataset(filepath, class_value):
    df = pd.read_csv(filepath, names=['name'])
    df = df['name'].str.split('/', expand=True)
    df.drop(columns=[0,1,2,3], axis=1, inplace=True)
    df.columns = ['bucket','name']
    df = df[['name', 'bucket']]
    df.insert(loc=0, column='class', value=class_value)
    return df

def plot_bar_chart(dataframe):
    pass


def autolabel(ax, rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 3, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def main():
    parser = argparse.ArgumentParser(
        description="visualize the splited dataset in different buckets"
    )
    parser.add_argument(
        "-I",
        "--iteration",
        help="Path to the folder where the input images are stored",
        type=int
    )
    args = parser.parse_args()

    global ITERATION
    ITERATION = args.iteration

    path = ROOT_DIR

    train_dataframe = []
    for file in os.listdir(path):
        if 'train' in file and file.endswith(".txt"):
            data = prepare_dataset(path+file, 'train')
            train_dataframe.append(data)
    train_dataframe = pd.concat(train_dataframe)

    val_dataframe = []
    for file in os.listdir(path):
        if 'validation' in file and file.endswith(".txt"):
            data = prepare_dataset(path+file, 'validation')
            val_dataframe.append(data)
    val_dataframe = pd.concat(val_dataframe)

    test_dataframe = []
    for file in os.listdir(path):
        if 'test' in file and file.endswith(".txt"):
            data = prepare_dataset(path+file, 'test')
            test_dataframe.append(data)
    test_dataframe = pd.concat(test_dataframe)


    labels, train_counts =  np.unique(train_dataframe.bucket, return_counts=True)
    labels, val_counts =  np.unique(val_dataframe.bucket, return_counts=True)
    labels, test_counts =  np.unique(test_dataframe.bucket, return_counts=True)

    x = np.arange(len(labels))
    width = 0.3  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, train_counts, width, color='#66b3ff', label='Train')
    rects2 = ax.bar(x , val_counts, width, color='#99ff99', label='Validation')
    rects3 = ax.bar(x + width, test_counts, width, color='#ffcc99', label='Test')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('No. of Images')
    ax.set_title('Splitted Dataset (Iteration: {0})'.format(ITERATION))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    autolabel(ax, rects1)
    autolabel(ax, rects2)
    autolabel(ax, rects3)

    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()




