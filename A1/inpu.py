import numpy as np
import math
import csv

def input():
    #All x0 = 1 -> IMP, np.zeros((N,2049)), y = np.zeros(N)
    
    x = []
    y = []
    with open('Assets/train.csv',newline='') as csvfile:
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
    with open('Assets/validation.csv',newline='') as csvfile:
        librarian = csv.reader(csvfile, delimiter= ' ', quotechar="|")
        for row in librarian:
            temp = row[0].split(",")
            yval.append(int(temp[1]))
            temp2 = [float(i) for i in temp[2:]]
            temp2.insert(0,1)
            xval.append(temp2)

    yval = np.array(yval)
    xval = np.array(xval)

    return N,x,y,xval,yval