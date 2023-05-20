import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from svm_multiclass import Trainer_OVA,Trainer_OVO
from kernel import rbf

actual = np.array([9,4,3,5,4,3,9,4,7,6,3,4,8,7,5,7,8,5,9,6,2,7,6,1,5,1,3,6,6,4,2,6,6,3,3,4,3,7,7])

'''
#C = 1 OVO
x = Trainer_OVO(rbf,1,10,gamma = 0.1)
x._init_trainers()
x.fit("Data sets/multi_train.csv")
main_test_labels = x.predict("Data sets/multi_val.csv")
print(main_test_labels)
confusion_matrix = metrics.confusion_matrix(actual, main_test_labels)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix,display_labels=[1,2,3,4,5,6,7,8,9,10])
cm_display.plot()
plt.show()

#C = 0.1 OVO
x = Trainer_OVO(rbf,0.1,10,gamma = 0.1)
x._init_trainers()
x.fit("Data sets/multi_train.csv")
main_test_labels = x.predict("Data sets/multi_val.csv")
print(main_test_labels)
confusion_matrix = metrics.confusion_matrix(actual, main_test_labels)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix,display_labels=[1,2,3,4,5,6,7,8,9,10])
cm_display.plot()
plt.show()
'''

#C = 1 OVA
x = Trainer_OVA(rbf,1,10,gamma = 0.1)
x._init_trainers()
x.fit("Data sets/multi_train.csv")
main_test_labels = x.predict("Data sets/multi_val.csv")
print(main_test_labels)
confusion_matrix = metrics.confusion_matrix(actual, main_test_labels,labels=[1,2,3,4,5,6,7,8,9,10])
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix,display_labels=[1,2,3,4,5,6,7,8,9,10])
cm_display.plot()
plt.show()

#C = 0.1 OVA
x = Trainer_OVA(rbf,0.1,10,gamma = 0.1)
x._init_trainers()
x.fit("Data sets/multi_train.csv")
main_test_labels = x.predict("Data sets/multi_val.csv")
print(main_test_labels)
confusion_matrix = metrics.confusion_matrix(actual, main_test_labels)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix,display_labels=[1,2,3,4,5,6,7,8,9,10])
cm_display.plot()
plt.show()

