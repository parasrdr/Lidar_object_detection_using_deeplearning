import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("precision_recall.csv")

iou_5 = df[df['IOU']== 0.5]

iou_5.sort_values(by =['recall'],inplace = True)

x = iou_5.recall
y = iou_5.precision
z = iou_5.confidence_threshold

mean_precision_iou5 = np.mean(y)
plt.plot(x,y,'r',label= 'IOU threshold:0.5')
plt.xlabel('recall')
plt.ylabel('precision')



iou_6 = df[df['IOU']== 0.6]

iou_6.sort_values(by =['recall'],inplace = True)
print(iou_6)
x = iou_6.recall
y = iou_6.precision
z = iou_6.confidence_threshold

mean_precision_iou6 = np.mean(y)
plt.plot(x,y,'g',label= 'IOU threshold:0.6')

iou_7 = df[df['IOU']== 0.7]

iou_7.sort_values(by =['recall'],inplace = True)

x = iou_7.recall
y = iou_7.precision
z = iou_7.confidence_threshold

mean_precision_iou7 = np.mean(y)
plt.plot(x,y,'b',label= 'IOU threshold:0.7')


plt.legend()
mean_AP = (mean_precision_iou5+ mean_precision_iou6+mean_precision_iou7)/3
mean_AP = round(mean_AP,3)
plt.title(f'Mean Average Precision mAP@IOU 0.5:0.1:0.7 = {mean_AP}')
plt.savefig('mAP.png',dpi =300)
