import numpy as np
import csv
import linear_fit as lf

def input():
    #All x0 = 1 -> IMP, np.zeros((N,2049)), y = np.zeros(N)
    diff = []
    for j in [2,5,10,100]:
        x2 = []
        y2 = []
        with open('Generalization/'+str(j)+'_d_train.csv',newline='') as csvfile:
            librarian = csv.reader(csvfile, delimiter= ' ', quotechar="|")
            for row in librarian:
                temp = row[0].split(",")
                y2.append(float(temp[-1]))
                temp2 = [float(i) for i in temp[:-1]]
                temp2.insert(0,1)
                x2.append(temp2)
        N = len(y2)
        y2 = np.array(y2)
        x2 = np.array(x2)
        w2 = np.zeros(j+1)

        alpha = 0.01
        it= 5000
        w2 = lf.linear_fit_maxit_noval(x2,y2,w2,N,alpha,it)
        w2 = w2[0]

        xtest = []
        ytest = []
        with open('Generalization/'+str(j)+'_d_test.csv',newline='') as csvfile:
            librarian = csv.reader(csvfile, delimiter= ' ', quotechar="|")
            for row in librarian:
                temp = row[0].split(",")
                ytest.append(float(temp[-1]))
                temp2 = [float(i) for i in temp[:-1]]
                temp2.insert(0,1)
                xtest.append(temp2)
        ytest = np.array(ytest)
        xtest = np.array(xtest)
        Ntest = len(ytest)

        Ein = np.sum((np.dot(x2,w2) - y2)**2)/N
        Eout = np.sum((np.dot(xtest,w2) - ytest)**2)/Ntest
        diff.append(Eout-Ein)
    print(diff)

def main():
    input()

if __name__ == "__main__":
    main()