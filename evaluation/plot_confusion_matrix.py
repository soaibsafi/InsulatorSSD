import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
array = [[427, 52],
        [0, 57]]
df_cm = pd.DataFrame(array, index = [i for i in "TF"],
                  columns = [i for i in "PN"])
sn.set(font_scale=1.4) # for label size
ax = sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='g') # font size
ax.xaxis.tick_top() # x axis on top
ax.xaxis.set_label_position('top')
ax.yaxis.tick_left() # x axis on top
ax.yaxis.set_label_position('left')
plt.show()