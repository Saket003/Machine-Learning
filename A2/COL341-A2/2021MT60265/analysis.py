from typing import List
from kernel import linear,rbf
import numpy as np
import qpsolvers
import csv
import svm_binary
import matplotlib.pyplot as plt

train_path = 'Data sets/bi_train.csv'
test_path = "Data sets/bi_val.csv"

print("Linear")
C_list = [0.01,0.1,1,10]
Ecv = [0,0,0,0]
#Linear case
for i in range(len(C_list)):
    c = C_list[i]
    x = svm_binary.Trainer(linear,c)
    x.fit(train_path)
    test_labels, Ecv[i]  = x.predict(test_path)     #TODO Uncomment Ecv from last line of predict() in svm_binary.py
    #You may search by comment in svm_binary.py
    print("C = ",c," ")
    for t in test_labels:
        print(t)
print(Ecv)

plt.plot(C_list,Ecv)
plt.title("Linear Kernel")
plt.xlabel("C")
plt.ylabel("Binary Cross-Validation Error")
plt.show()

C_list = [0.01,0.1,1,10]
gamma_list = [0.1,0.01,0.001]
Ecv= [[0,0,0],[0,0,0],[0,0,0],[0,0,0]] #First indices by C, then by gamma
for i in range(len(C_list)):
    c = C_list[i]
    for j in range(len(gamma_list)):
        g = gamma_list[j]
        x = svm_binary.Trainer(rbf,c,gamma=g)
        x.fit(train_path)
        test_labels, Ecv[i][j]  = x.predict(test_path)      #TODO Uncomment Ecv from line 91 in svm_binary.py
        print("C = ",c," Gamma = ",g," ")
        for t in test_labels:
            print(t)
print(Ecv)

#For fixed C,
for i in range(len(C_list)):
    plt.plot(gamma_list,Ecv[i])
    plt.title("RBF Kernel for fixed C")
    plt.xlabel("\u03B3")
    plt.ylabel("Binary Cross-Validation Error")
plt.legend(C_list)
plt.show()

#For fixed gamma,
for i in range(len(gamma_list)):
    l = [Ecv[j][i] for j in range(4)]
    plt.plot(C_list,l)
    plt.title("RBF Kernel for fixed \u03B3")
    plt.xlabel("C")
    plt.ylabel("Binary Cross-Validation Error")
plt.legend(gamma_list)
plt.show()