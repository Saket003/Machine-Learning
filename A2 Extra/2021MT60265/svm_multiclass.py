from typing import List
import numpy as np
from svm_binary import Trainer
from kernel import linear,rbf,laplacian,sigmoid,polynomial

class Trainer_OVO:
    def __init__(self, kernel, C=None, n_classes = -1, **kwargs) -> None:
        self.kernel = kernel
        self.C = C
        self.n_classes = n_classes
        self.kwargs = kwargs
        self.svms = [] # List of Trainer objects [Trainer]
    
    def _init_trainers(self):
        count = int(self.n_classes*(self.n_classes-1)/2)
        for i in range(count):
            self.svms.append(Trainer(rbf,self.C,**self.kwargs))
    
    def fit(self, train_data_path:str, max_iter=None)->None:
        count = int(self.n_classes*(self.n_classes-1)/2)
        k = 0
        for i in range(self.n_classes):
            for j in range(i+1,self.n_classes):     
                self.svms[k].fit3(train_data_path,i+1,j+1)
                k = k+1
    
    def predict(self, test_data_path:str)->np.ndarray:
        all_test_labels = []
        count = int(self.n_classes*(self.n_classes-1)/2)
        k = 0
        for i in range(self.n_classes):
            for j in range(i+1,self.n_classes):     
                all_test_labels.append(self.svms[k].predict3(test_data_path,i+1,j+1))
                k = k+1
        
        main_test_labels = []
        j_range = len(all_test_labels[0])
        most_predicted_classes = []
        for j in range(j_range):
            for i in range(len(all_test_labels)):
                most_predicted_classes.append(all_test_labels[i][j])
            most_repeated = self.max_occur(most_predicted_classes)
            most = most_repeated[0]
            if(len(most_repeated) == 1):
                most = most_repeated[0]
            else:
                most = self.tiebreaker(most_repeated,j,test_data_path)
            main_test_labels.append(most)
            most_predicted_classes = []

        main_test_labels = np.array(main_test_labels)
        return main_test_labels
    
    def max_occur(self,list_classes):
        list_classes.sort()
        num_list = []
        max_times = 0
        times = 1
        for i in range(1,len(list_classes)):
            if((list_classes[i]==list_classes[i-1])):
                times +=1
            else:
                if(times == max_times):
                    num_list.append(list_classes[i-1])
                elif(times > max_times):
                    num_list = []
                    num_list.append(list_classes[i-1])
                    max_times = times
                times = 1
        if(times == max_times):
            num_list.append(list_classes[i-1])
        elif(times > max_times):
            num_list = []
            num_list.append(list_classes[i-1])
            max_times = times
        times = 1
        return num_list
    
    def tiebreaker(self,list_most,j,test_data_path):
        list_most.sort()
        most = list_most[0]
        for i in range(1,len(list_most)):
            a = most
            b = list_most[i]
            index = self.n_classes*(a-1) - a*a + a + b - 1
            svm = self.svms[index]
            label = svm.predict3(test_data_path,a,b)[j]
            if(label == b):
                most = b
        return most


class Trainer_OVA:
    def __init__(self, kernel, C=None, n_classes = -1, **kwargs) -> None:
        self.kernel = kernel
        self.C = C
        self.n_classes = n_classes
        self.kwargs = kwargs
        self.svms = [] # List of Trainer objects [Trainer]
    
    def _init_trainers(self):
        for i in range(self.n_classes):
            self.svms.append(Trainer(rbf,self.C,**self.kwargs))
    
    def fit(self, train_data_path:str, max_iter=None)->None:
        for i in range(self.n_classes):
            self.svms[i].fit2(train_data_path,i+1)
    
    def predict(self, test_data_path:str)->np.ndarray:
        all_test_labels = []
        for i in range(self.n_classes):
            all_test_labels.append(self.svms[i].predict2(test_data_path,i+1))
        
        main_test_labels = []
        j_range = len(all_test_labels[0])
        for j in range(j_range):
            max = all_test_labels[0][j]
            max_i = 1
            for i in range(len(all_test_labels)):
                if(all_test_labels[i][j] > max):
                    max = all_test_labels[i][j]
                    max_i = i+1
            main_test_labels.append(max_i)

        main_test_labels = np.array(main_test_labels)
        return main_test_labels

'''

print("C = 1 OVA")
#C = 1 OVA
x = Trainer_OVA(rbf,1,10,gamma = 0.1)
x._init_trainers()
x.fit("Data sets/multi_train.csv")
main_test_labels = x.predict("Data sets/multi_val.csv")

for i in range(len(main_test_labels)):
    print(main_test_labels[i])
    pass

print("C = 0.1 OVA")
#C = 0.1 OVA
x = Trainer_OVA(rbf,0.1,10,gamma = 0.1)
x._init_trainers()
x.fit("Data sets/multi_train.csv")
main_test_labels = x.predict("Data sets/multi_val.csv")

for i in range(len(main_test_labels)):
    print(main_test_labels[i])
    pass

    
print("C = 1 OVO")
#C = 1 OVO
x = Trainer_OVO(rbf,1,10,gamma = 0.1)
x._init_trainers()
x.fit("Data sets/multi_train.csv")
main_test_labels = x.predict("Data sets/multi_val.csv")

for i in range(len(main_test_labels)):
    print(main_test_labels[i])
    pass


print("C = 0.1 OVO")
#C = 0.1 OVO
x = Trainer_OVO(rbf,0.1,10,gamma = 0.1)
x._init_trainers()
x.fit("Data sets/multi_train.csv")
main_test_labels = x.predict("Data sets/multi_val.csv")

for i in range(len(main_test_labels)):
    print(main_test_labels[i])
    pass

'''
'''
x = Trainer_OVO(rbf,1,10,gamma = 0.1)
l = x.max_occur([4,1,4,9,8,9])
'''