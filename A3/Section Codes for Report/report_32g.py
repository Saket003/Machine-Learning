from TakeInput import take_input_multiclass
import time
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from confusion import accuracy

train_path = "data/train"
feature_vector, output_vector = take_input_multiclass(train_path,2000)
validation_path = "data/validation"
feature_val, output_val = take_input_multiclass(validation_path,400)

begin = time.time()
clf = XGBClassifier(random_state = 0,max_depth=10,n_estimators=50,subsample=0.6)
clf.fit(feature_vector,output_vector)
end = time.time()
print(f"Time taken is {end - begin} seconds")

accuracy(feature_vector,output_vector,clf)
accuracy(feature_val,output_val,clf)

n_samples = 12
feature_val = np.zeros(shape = (n_samples, 3072))

i = 0
for file in os.listdir("3.2G Testing"):
    img = cv2.imread("3.2G Testing" +"/" +file)
    feature_val[i] = img.reshape(1,3072)
    i +=1

output_val = np.ones(n_samples)
predicted_val = clf.predict(feature_val)
confusion_matrix = metrics.confusion_matrix(output_val, predicted_val,labels=[0,1,2,3])
sum = 0
for i in range(4):
    sum = sum + confusion_matrix[i][i]
accuracy = (sum)/len(output_val)
print(accuracy)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix,display_labels=[0,1,2,3])
cm_display.plot()
plt.show()