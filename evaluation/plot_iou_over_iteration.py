import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

  
# Averyge Precision
iteration = [1, 2, 3, 4, 5, 6, 7, 8]
iou50_95 =          [0.405, 0.429, 0.471, 0.467, 0.469, 0.459, 0.444, 0.429]
iou50 =             [0.765, 0.784, 0.814, 0.816, 0.830, 0.833, 0.823, 0.837]
iou75 =             [0.400, 0.425, 0.503, 0.464, 0.462, 0.436, 0.428, 0.398]
iou50_95_small =    [0.270, 0.279, 0.352, 0.335, 0.344, 0.312, 0.303, 0.308]
iou50_95_medium =   [0.585, 0.622, 0.619, 0.644, 0.645, 0.631, 0.627, 0.585]
iou50_95_large =    [0.739, 0.817, 0.792, 0.814, 0.774, 0.720, 0.714, 0.634]

plt.gcf().clear()
fig = plt.figure(1)

ax = fig.add_subplot(111)
ax.plot(iteration, iou50_95, label='IoU=0.50:0.95')
ax.plot(iteration, iou50, label='IoU=0.50')
ax.plot(iteration, iou75, label='IoU=0.75')
ax.plot(iteration, iou50_95_small, label='IoU=0.50:0.95-Small')
ax.plot(iteration, iou50_95_medium, label='IoU=0.50:0.95-Medium')
ax.plot(iteration, iou50_95_large, label='IoU=0.50:0.95-Large')

handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
text = ax.text(-0.2,1.05, "", transform=ax.transAxes)
ax.set_title("Average Precision over Iteration")
plt.xlabel('Iteration')
plt.ylabel('Average Precision (AP)')
#plt.show()
fig.savefig('Average_Precision_over_iteration', bbox_extra_artists=(lgd,text), bbox_inches='tight', dpi=300)


# Averyge Recall
iteration = [1, 2, 3, 4, 5, 6, 7, 8]
ar_iou50_95 =          [0.207, 0.219, 0.218, 0.231, 0.232, 0.221, 0.228, 0.214]
ar_iou50 =             [0.447, 0.477, 0.523, 0.521, 0.524, 0.515, 0.492, 0.483]
ar_iou75 =             [0.491, 0.517, 0.555, 0.557, 0.557, 0.548, 0.538, 0.527]
ar_iou50_95_small =    [0.381, 0.383, 0.450, 0.441, 0.443, 0.441, 0.430, 0.438]
ar_iou50_95_medium =   [0.642, 0.690, 0.685, 0.708, 0.710, 0.698, 0.691, 0.647]
ar_iou50_95_large =    [0.739, 0.848, 0.834, 0.847, 0.825, 0.814, 0.759, 0.713]


plt.gcf().clear()
fig = plt.figure(1)

ax = fig.add_subplot(111)
ax.plot(iteration, ar_iou50_95, label='IoU=0.50:0.95')
ax.plot(iteration, ar_iou50, label='IoU=0.50')
ax.plot(iteration, ar_iou75, label='IoU=0.75')
ax.plot(iteration, ar_iou50_95_small, label='IoU=0.50:0.95-Small')
ax.plot(iteration, ar_iou50_95_medium, label='IoU=0.50:0.95-Medium')
ax.plot(iteration, ar_iou50_95_large, label='IoU=0.50:0.95-Large')

handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
text = ax.text(-0.2,1.05, "", transform=ax.transAxes)
ax.set_title("Average Recall over Iteration")
plt.xlabel('Iteration')
plt.ylabel('Average Recall (AR)')
#plt.show()
fig.savefig('Average_Recall_over_iteration', bbox_extra_artists=(lgd,text), bbox_inches='tight', dpi=300)