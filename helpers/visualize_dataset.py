import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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
    train_dataframe = prepare_dataset('data/train_20220421-131254.txt', 'train')
    val_dataframe = prepare_dataset('data/validation_20220421-131254.txt', 'validation')
    test_dataframe = prepare_dataset('data/test_20220421-131254.txt', 'test')

    labels, train_counts =  np.unique(train_dataframe.bucket, return_counts=True)
    labels, val_counts =  np.unique(val_dataframe.bucket, return_counts=True)
    labels, test_counts =  np.unique(test_dataframe.bucket, return_counts=True)

    x = np.arange(len(labels))
    width = 0.3  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, train_counts, width, label='Train')
    rects2 = ax.bar(x , val_counts, width, label='Validation')
    rects3 = ax.bar(x + width, test_counts, width, label='Test')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('No. of Images')
    ax.set_title('Splitting the dataset')
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




