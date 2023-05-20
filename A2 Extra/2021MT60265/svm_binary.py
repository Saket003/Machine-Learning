from typing import List
import numpy as np
import qpsolvers
import csv
from kernel import linear,rbf,laplacian,sigmoid,polynomial

class Trainer:
    def __init__(self,kernel,C=None,**kwargs) -> None:
        self.kernel_fn = kernel
        self.kernel = 0
        self.kwargs = kwargs
        self.C=C
        self.support_vectors:List[np.ndarray] = []
        self.support_indices = []
        self.b = 0
        self.y_train = []
        self.alpha_train = []
        self.x_train = []
        
    
    def fit(self, train_data_path:str)->None:
        x = []
        y = []
        x,y,N_samples = self.input1(x,y,train_data_path)
        
        self.kernel = self.kernel_fn(x,**self.kwargs)

        P = np.zeros_like(self.kernel)
        for i in range(0,N_samples):
            for j in range(0,N_samples):
                P[i][j] = y[i]*y[j]*(self.kernel[i][j])
        q = np.zeros((1,N_samples)) - 1
        lb = np.zeros((1,N_samples))
        ub = np.zeros((1,N_samples)) + self.C
        A = y.copy()   
        A.reshape(1,-1)
        B = np.array([0.0])
        alpha = qpsolvers.solve_qp(P,q,None,None,A,B,lb,ub,solver="ecos") 
        self.alpha_train = alpha
        self.y_train = y
        self.x_train = x

        if(type(alpha) != np.ndarray):
            print("No alpha")
            return

        thres = 1e-10 
        for i in range (N_samples):
            if(alpha[i]>thres and alpha[i]<self.C):
                self.support_indices.append(i)
                self.support_vectors.append(x[i])
        for i in range (N_samples):
            if(alpha[i] == self.C):
                self.support_indices.append(i)
                self.support_vectors.append(x[i])
        
        
        
        s = 0
        while(True):
            if(alpha[s]>thres and alpha[s]<self.C): 
                break
            s = s + 1
        b = y[s]
        for i in range(N_samples):
            b = b - y[i]*alpha[i]*self.kernel[i][s]
        self.b = b


    def predict(self, test_data_path:str)->np.ndarray:
        x_test = []
        y_test= []

        if(type(self.alpha_train) != np.ndarray):
            return None
        
        x_test,y_test,N_test = self.input1(x_test,y_test,test_data_path)
        test_labels = [0 for i in range(N_test)]

        for i in range(N_test):
            label = self.b
            for j in self.support_indices:
                X = np.array([self.x_train[j],x_test[i]])
                mini_K = self.kernel_fn(X,**self.kwargs)
                label = label + self.y_train[j]*self.alpha_train[j]*mini_K[0][1]
            test_labels[i] = 1 if label > 0 else -1
        
        misclassified = 0
        for i in range(len(y_test)):
            if(y_test[i]!=test_labels[i]):
                misclassified +=1
        Ecv = misclassified/len(y_test)

        test_labels = np.array(test_labels)
        return test_labels#,Ecv 
        #Uncomment out for analysis.py
    
    def input1(self,x,y,path):
        first = 0
        with open(path,newline='') as csvfile:
            librarian = csv.reader(csvfile, delimiter= ' ', quotechar="|")
            for row in librarian:
                if(first == 0):
                    first = first + 1
                    continue
                temp = row[0].split(",")
                y.append(1 if int(float(temp[len(temp)-1]))==1 else -1)
                temp2 = [float(i) for i in temp[1:len(temp)-1]]
                temp2 = np.array(temp2)
                x.append(temp2)
        N_samples = len(y)
        y = np.array(y)
        x = np.array(x)
        return x,y,N_samples
    

    '''
    For OVA - Modifications mostly due to inconsistent input format, and probability rather
    than sign requirement
    '''
    def fit2(self, train_data_path:str, target)->None:
        x = []
        y = []
        x,y,N_samples = self.input_ova(x,y,train_data_path,target)
        
        self.kernel = self.kernel_fn(x,**self.kwargs)


        P = np.zeros_like(self.kernel)
        for i in range(0,N_samples):
            for j in range(0,N_samples):
                P[i][j] = y[i]*y[j]*(self.kernel[i][j])
        q = np.zeros((1,N_samples)) - 1
        lb = np.zeros((1,N_samples))
        ub = np.zeros((1,N_samples)) + self.C
        A = y.copy()
        A.reshape(1,-1)
        B = np.array([0.0])
        alpha = qpsolvers.solve_qp(P,q,None,None,A,B,lb,ub,solver="ecos")
        self.alpha_train = alpha
        self.y_train = y
        self.x_train = x

        if(type(alpha) != np.ndarray):
            print("No alpha")
            return

        thres = 1e-10 
        for i in range (N_samples):
            if(alpha[i]>thres and alpha[i]<self.C):
                self.support_indices.append(i)
                self.support_vectors.append(x[i])
        for i in range (N_samples):
            if(alpha[i] == self.C):
                self.support_indices.append(i)
                self.support_vectors.append(x[i])
        
        
        
        s = 0
        while(True):
            if(alpha[s]>thres and alpha[s]<self.C): 
                break
            s = s + 1
        b = y[s]
        for i in range(N_samples):
            b = b - y[i]*alpha[i]*self.kernel[i][s]
        self.b = b


    def predict2(self, test_data_path:str,target)->np.ndarray:
        x_test = []
        y_test= []

        if(type(self.alpha_train) != np.ndarray):
            return None
        
        x_test,y_test,N_test = self.input_ova(x_test,y_test,test_data_path,target)
        test_labels = [0 for i in range(N_test)]

        for i in range(N_test):
            label = self.b
            for j in self.support_indices:
                X = np.array([self.x_train[j],x_test[i]])
                mini_K = self.kernel_fn(X,**self.kwargs)
                label = label + self.y_train[j]*self.alpha_train[j]*mini_K[0][1]
            test_labels[i] = label
        
        test_labels = np.array(test_labels)
        return test_labels
        
    def input_ova(self,x,y,path,target):
        first = 0

        with open(path,newline='') as csvfile:
            librarian = csv.reader(csvfile, delimiter= ' ', quotechar="|")
            for row in librarian:
                if(first == 0):
                    first = first + 1
                    continue
                temp = row[0].split(",")

                y.append(1 if int(float(temp[1]))==target else -1)
                temp2 = [float(i) for i in temp[2:len(temp)]]

                temp2 = np.array(temp2)
                x.append(temp2)
        
        N_samples = len(y)
        y = np.array(y)
        x = np.array(x)
        return x,y,N_samples
    


    def fit3(self, train_data_path:str, c1,c2)->None:
        x = []
        y = []
        x,y,N_samples = self.input_ovo(x,y,train_data_path,c1,c2)
        
        self.kernel = self.kernel_fn(x,**self.kwargs)


        P = np.zeros_like(self.kernel)
        for i in range(0,N_samples):
            for j in range(0,N_samples):
                P[i][j] = y[i]*y[j]*(self.kernel[i][j])
        q = np.zeros((1,N_samples)) - 1
        lb = np.zeros((1,N_samples))
        ub = np.zeros((1,N_samples)) + self.C
        A = y.copy()   
        A.reshape(1,-1)
        B = np.array([0.0])
        alpha = qpsolvers.solve_qp(P,q,None,None,A,B,lb,ub,solver="ecos")
        self.alpha_train = alpha
        self.y_train = y
        self.x_train = x

        if(type(alpha) != np.ndarray):
            print("No alpha")
            return

        thres = 1e-10 
        for i in range (N_samples):
            if(alpha[i]>thres and alpha[i]<self.C):
                self.support_indices.append(i)
                self.support_vectors.append(x[i])
        for i in range (N_samples):
            if(alpha[i] == self.C):
                self.support_indices.append(i)
                self.support_vectors.append(x[i])
        
        
        
        s = 0
        while(True):
            if(alpha[s]>thres and alpha[s]<self.C): 
                break
            s = s + 1
        b = y[s]
        for i in range(N_samples):
            b = b - y[i]*alpha[i]*self.kernel[i][s]
        self.b = b


    def predict3(self, test_data_path:str,c1,c2)->np.ndarray:
        x_test = []
        y_test= []

        if(type(self.alpha_train) != np.ndarray):
            return None
        
        x_test,y_test,N_test = self.input_ovo(x_test,y_test,test_data_path,c1,c2)
        test_labels = [0 for i in range(N_test)]

        for i in range(N_test):
            label = self.b
            for j in self.support_indices:
                X = np.array([self.x_train[j],x_test[i]])
                mini_K = self.kernel_fn(X,**self.kwargs)
                label = label + self.y_train[j]*self.alpha_train[j]*mini_K[0][1]
            test_labels[i] = c1 if label > 0 else c2
        
        test_labels = np.array(test_labels)
        return test_labels
        
    def input_ovo(self,x,y,path,c1,c2):
        first = 0

        with open(path,newline='') as csvfile:
            librarian = csv.reader(csvfile, delimiter= ' ', quotechar="|")
            for row in librarian:
                if(first == 0):
                    first = first + 1
                    continue
                temp = row[0].split(",")

                y.append(1 if int(float(temp[1]))==c1 else -1)
                temp2 = [float(i) for i in temp[2:len(temp)]]

                temp2 = np.array(temp2)
                x.append(temp2)
        
        N_samples = len(y)
        y = np.array(y)
        x = np.array(x)
        return x,y,N_samples
    
'''
x = Trainer(linear,10)
x.fit('Data sets/bi_train.csv')
print(x.b)
test_labels = x.predict("Data sets/bi_val.csv")
print(test_labels)
'''
'''
x = Trainer(linear,10)
x.fit('Data sets/bi_train.csv')
test_labels,Ecv  = x.predict("Data sets/bi_val.csv")
print(x.b)
print(Ecv*100)
'''

'''
Passed - 
x = Trainer(rbf,1,gamma=0.1)
x.fit('Data sets/bi_train.csv')
print(x.b)
test_labels = x.predict("Data sets/bi_val.csv")
print(test_labels)

Not PSD error -
x = Trainer(sigmoid,1,gamma = 5,r = 1)
x.fit('Data sets/bi_train.csv')
print(x.b)
test_labels = x.predict("Data sets/bi_val.csv")
print(test_labels)

x = Trainer(polynomial,1,zeta=0,gamma=1,Q=1)
x.fit('Data sets/bi_train.csv')
print(x.b)
test_labels = x.predict("Data sets/bi_val.csv")
print(test_labels)

x = Trainer(laplacian,1,gamma = 0.01)
x.fit('Data sets/bi_train.csv')
print(x.b)
test_labels = x.predict("Data sets/bi_val.csv")
print(test_labels)


#TODO
Runs but not find alpha(Search direction unreliable) - 
x = Trainer(polynomial,1,zeta=5,gamma=2,Q=2)
x.fit('Data sets/bi_train.csv')
print(x.b)
test_labels = x.predict("Data sets/bi_val.csv")
print(test_labels)
'''

