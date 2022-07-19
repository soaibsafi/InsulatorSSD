import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

  
# Averyge Precision
iteration = [1, 2, 3, 4, 5, 6, 7, 8]
iou50_95 =          [0.741, 0.765, 0.778, 0.793, 0.789, 0.815, 0.833, 0.859]
iou50 =             [0.822, 0.851, 0.861, 0.877, 0.891, 0.914, 0.916, 0.945]
iou75 =             [0.719, 0.732, 0.701, 0.724, 0.736, 0.729, 0.725, 0.744]

iou50_95_small =    [0.617, 0.621, 0.635, 0.613, 0.649, 0.663, 0.650, 0.674]
iou50_95_medium =   [0.815, 0.822, 0.819, 0.834, 0.825, 0.831, 0.847, 0.840]
iou50_95_large =    [0.909, 0.917, 0.912, 0.914, 0.924, 0.920, 0.914, 0.919]

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
ar_iou50_95 =          [0.607, 0.619, 0.618, 0.621, 0.632, 0.621, 0.628, 0.634]
ar_iou50 =             [0.747, 0.757, 0.743, 0.761, 0.764, 0.775, 0.782, 0.773]


ar_iou75 =             [0.901, 0.917, 0.905, 0.919, 0.924, 0.928, 0.938, 0.927]



ar_iou50_95_small =    [0.641, 0.653, 0.640, 0.641, 0.648, 0.641, 0.637, 0.638]
ar_iou50_95_medium =   [0.742, 0.750, 0.745, 0.758, 0.760, 0.773, 0.779, 0.782]
ar_iou50_95_large =    [0.859, 0.858, 0.844, 0.857, 0.835, 0.844, 0.852, 0.861]


plt.gcf().clear()
fig = plt.figure(1)

ax = fig.add_subplot(111)
ax.plot(iteration, ar_iou50_95, label='IoU=0.50:0.95; maxDets=1')
ax.plot(iteration, ar_iou50, label='IoU=0.50:0.95; maxDets=10')
ax.plot(iteration, ar_iou75, label='IoU=0.50:0.95; maxDets=100')
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