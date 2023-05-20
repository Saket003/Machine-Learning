import numpy as np
import csv
import linear_fit as lf
import sys

def inpu(train_path,val_path,test_path):
    #All x0 = 1 -> IMP, np.zeros((N,2049)), y = np.zeros(N)
    x = []
    y = []
    with open(train_path,newline='') as csvfile:
        librarian = csv.reader(csvfile, delimiter= ' ', quotechar="|")
        for row in librarian:
            temp = row[0].split(",")
            y.append(int(temp[1]))
            temp2 = [float(i) for i in temp[2:]]
            temp2.insert(0,1)
            x.append(temp2)
    N = len(y)
    y = np.array(y)
    x = np.array(x)


    xval = []
    yval = []
    with open(val_path,newline='') as csvfile:
        librarian = csv.reader(csvfile, delimiter= ' ', quotechar="|")
        for row in librarian:
            temp = row[0].split(",")
            yval.append(int(temp[1]))
            temp2 = [float(i) for i in temp[2:]]
            temp2.insert(0,1)
            xval.append(temp2)
    yval = np.array(yval)
    xval = np.array(xval)

    xtest = []
    with open(test_path,newline='') as csvfile:
        librarian = csv.reader(csvfile, delimiter= ' ', quotechar="|")
        for row in librarian:
            temp = row[0].split(",")
            temp2 = [float(i) for i in temp[1:]]
            temp2.insert(0,1)
            xtest.append(temp2)
    xtest = np.array(xtest)
    return N,x,y,xval,yval,xtest


def output(test_path,ytest,out_path):
    sample_name = []
    with open(test_path,newline='') as csvfile:
        librarian = csv.reader(csvfile, delimiter= ' ', quotechar="|")
        for row in librarian:
            temp = row[0].split(",")
            sample_name.append(temp[0])

    with open(out_path, 'w', newline='') as csvfile:
        swriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range (len(ytest)):
            swriter.writerow([sample_name[i], ytest[i]])


def multi_class_fit(x,y,W,N,alpha,it):
    grad = []
    for i in range (0,8):
        grad.append(np.zeros(W[0].size))

    while(it > 0):
        it = it - 1
        
        grad = []
        for i in range (0,8):
            grad.append(np.zeros(W[0].size))
        
        for j in range(N):
            Dr = 1
            for i in range(0,8):
                Dr = Dr + np.exp(np.dot(x[j],W[i]))
            for i in range(0,8):
                y_bool = 1 if y[j] == i+1 else 0
                grad[i] = grad[i] + ((np.exp(np.dot(x[j],W[i]))/Dr)-y_bool)*(x[j])
        
        for i in range(0,8):
            W[i] = W[i] - alpha*grad[i]
    
    return W


def main(train_path,val_path,test_path,out_path,section):
    N,x,y,xval,yval,xtest = inpu(train_path,val_path,test_path)
    #1 - Linear
    if(section == '1'):
        alpha = 0.001
        bound = 0.000001
        w = np.zeros(int(x.size/N))
        w = lf.linear_fit_reltol(x,y,xval,yval,w,N,alpha,bound)[0]
        ytest = np.dot(xtest,w)
        
    #2 - Ridge
    elif(section == '2'):
        alpha = 0.001
        it= 1000
        l = 15
        w = np.zeros(int(x.size/N))
        w = lf.ridge_fit_maxit(x,y,xval,yval,w,N,alpha,it,l)[0]
        ytest = np.dot(xtest,w)

    #5 - Classification
    else:
        W = []
        for i in range (0,8):
            W.append(np.zeros(2049))
        alpha = 0.01
        it = 1000
        W = multi_class_fit(x,y,W,N,alpha,it)

        ytest = np.zeros(len(xtest))
        for i in range(len(xtest)):
            Dr = 1
            for j in range(0,8):
                Dr = Dr + np.exp(np.dot(xtest[i],W[j]))
            
            prob = [0,0,0,0,0,0,0,0,0]
            for j in range(8):
                prob[j] = np.exp(np.dot(xtest[i],W[j]))/Dr
            prob[8] = 1 - prob[0] - prob[1] - prob[2] - prob[3] - prob[4] - prob[5] - prob[6] - prob[7]

            max_index = prob.index(max(prob))
            ytest[i] = max_index
        
    output(test_path,ytest,out_path)
    

if __name__ == "__main__":
    main(sys.argv[1].split("=")[1],sys.argv[2].split("=")[1],sys.argv[3].split("=")[1],sys.argv[4].split("=")[1],sys.argv[5].split("=")[1])

"""
python main.py --train_path=.\Assets\train.csv --val_path=.\Assets\validation.csv --test_path=.\Assets\test.csv --out_path=.\Outputs\Out1.csv --section=1
"""
