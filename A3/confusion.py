import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from TakeInput import take_input
from sklearn import tree
import time
from sklearn import metrics

def confusion(feature_vector,output_vector,clf):
    predicted_val = clf.predict(feature_vector)
    confusion_matrix = metrics.confusion_matrix(output_vector, predicted_val,labels=[0,1])
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix,display_labels=[0,1])
    cm_display.plot()
    plt.show()

    accuracy = (confusion_matrix[0][0] + confusion_matrix[1][1])/len(output_vector)
    #print("Accuracy = ",accuracy)
    #print("Precision of 1 = ",(confusion_matrix[0][0]/(confusion_matrix[0][0] + confusion_matrix[1][0])))
    #print("Precision of 0 = ",(confusion_matrix[1][1]/(confusion_matrix[1][1] + confusion_matrix[0][1])))
    #print("Recall of 1 = ",(confusion_matrix[0][0]/(confusion_matrix[0][0] + confusion_matrix[0][1])))
    #print("Recall of 0 = ",(confusion_matrix[1][1]/(confusion_matrix[1][1] + confusion_matrix[1][0])))

    print(accuracy)
    print((confusion_matrix[0][0]/(confusion_matrix[0][0] + confusion_matrix[1][0])))
    print((confusion_matrix[1][1]/(confusion_matrix[1][1] + confusion_matrix[0][1])))
    print((confusion_matrix[0][0]/(confusion_matrix[0][0] + confusion_matrix[0][1])))
    print((confusion_matrix[1][1]/(confusion_matrix[1][1] + confusion_matrix[1][0])))
    print()

    
def accuracy(feature_vector,output_vector,clf):
    predicted_val = clf.predict(feature_vector)
    confusion_matrix = metrics.confusion_matrix(output_vector, predicted_val,labels=[0,1,2,3])
    
    sum = 0
    for i in range(4):
        sum = sum + confusion_matrix[i][i]
    accuracy = (sum)/len(output_vector)
    print(accuracy)
    print()

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix,display_labels=[0,1,2,3])
    cm_display.plot()
    plt.show()