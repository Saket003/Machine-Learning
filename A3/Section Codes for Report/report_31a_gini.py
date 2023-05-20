from TakeInput import take_input
import time
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

def predict(root,feature_val,size):
    predicted_val = np.zeros(size)
    i = 0
    safe_root = root
    for features in feature_val:
        root = safe_root
        while(True):
            if(root.isLeaf == True):
                predicted_val[i] = root.leafvalue
                i += 1
                break
            root = root.children[0 if int(features[root.para_index]) < root.threshold else 1]
    return predicted_val

class Node:
    def __init__(self,para_index=None):
        self.para_index = para_index
        self.isLeaf = False
        self.leafvalue = None
        self.children = [None, None]
        self.threshold = 0
        self.parent = None

class DecisionTree:
    def __init__(self,max_depth,min_samples_split,feature_vector,output_vector):
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def leafval(self, root, feature_vector, output_vector,bool_vector):
        if(root.isLeaf == True):
            zeros = 0
            ones = 0
            for i in range(len(bool_vector)):
                if(bool_vector[i]==0):
                    continue
                else:
                    if(output_vector[i]==0):
                        zeros+=1
                    else:
                        ones+=1
            root.leafvalue = 1 if(3*ones>zeros) else 0

        else:
            thres = root.threshold
            para = root.para_index
            bool_1 = np.copy(bool_vector)
            bool_2 = np.copy(bool_vector)
            for i in range(len(bool_vector)):
                if(feature_vector[i,para]>thres):
                    bool_1[i] = 0
                else:
                    bool_2[i] = 0
            self.leafval(root.children[0],feature_vector,output_vector,bool_1)
            self.leafval(root.children[1],feature_vector,output_vector,bool_2)
        return

    def ginis(self, indices, feature_vector, output_vector):
        vector = []
        for w in range(indices):
            max_avg_gini = 0
            temp = 0

            column = np.copy(feature_vector[:,w])
            sort = np.unravel_index(np.argsort(column, axis=None), column.shape)
            column.sort()
            output_sorted_with_column = output_vector[sort]

            thresholds = []
            for i in range(len(column)-5):
                if(output_sorted_with_column[i]!=output_sorted_with_column[i+3]):
                    if(output_sorted_with_column[i]==output_sorted_with_column[i+1] and output_sorted_with_column[i+2]==output_sorted_with_column[i+1] and output_sorted_with_column[i+3]==output_sorted_with_column[i+5] and output_sorted_with_column[i+4]==output_sorted_with_column[i+5]):
                        thresholds.append((column[i]+column[i+1])/2)

            it = 1
            if(len(thresholds) == 0):
                max_avg_gini = float("inf")
                temp = 256

            for thres in thresholds:
                ones0 = 0
                ones1 = 0
                zeros0 = 0
                zeros1 = 0
                total0 = 0
                total1 = 1
                i = 0
                while(i < len(column)):
                    if(column[i] > thres):
                        break
                    if(output_sorted_with_column[i] == 0):
                        zeros0 += 1
                    total0 += 1
                    i += 1
                ones0 = total0 - zeros0
                while(i < len(column)):
                    if(output_sorted_with_column[i] == 0):
                        zeros1 += 1
                    total1 += 1
                    i += 1
                ones1 = total1 - zeros1
                t1 = (ones0/total0)**2#Check correct power?
                t2 = (zeros0/total0)**2
                t3 = (ones1/total1)**2
                t4 = (zeros1/total1)**2
                avg_gini = (2-t1-t2-t3-t4)/2#Check

                if(avg_gini > max_avg_gini or it == 1):
                    max_avg_gini = avg_gini
                    temp = thres
                    it = 2
            vector.append(list([max_avg_gini,temp]))
        return vector

    def ID3(self,depth,feature_vector, output_vector, parent_gini):
        root = Node()
        max_threshold = 0
        gain_max = -1
        k = 0
        self.info = self.ginis(len(feature_vector[0]),feature_vector,output_vector)

        for w in range(len(feature_vector)):
            max_avg_gini = self.info[w][0]
            temp = self.info[w][1]
            gain = parent_gini - max_avg_gini
            if(gain > gain_max):
                k = w
                max_threshold = temp
                gain_max = gain

        root.para_index = k
        root.threshold = max_threshold

        left_child = Node()
        total_left = np.count_nonzero(feature_vector[:,k] < max_threshold)
        if(total_left < self.min_samples_split or depth == self.max_depth-1):
            left_child.isLeaf = True
        else:
            zeros0 = 0
            total0 = 0
            for i in range(len(feature_vector[:,k])):
                val = feature_vector[i,k]
                if(val>max_threshold):
                    continue
                if(output_vector[i] == 0):
                    zeros0 += 1
                total0 += 1
            ones0 = total0 - zeros0
            t1 = (ones0/total0)**2#Check correct power?
            t2 = (zeros0/total0)**2
            
            feature_cap = np.zeros((total0,len(feature_vector[0])))
            output_cap = np.zeros(total0)
            m = 0
            for i in range(len(feature_vector[:,k])):
                val = feature_vector[i,k]
                if(val>max_threshold):
                    continue
                feature_cap[m] = feature_vector[i]
                output_cap[m] = output_vector[i]
                m += 1
            
            left_child = self.ID3(depth+1,feature_cap,output_cap,1-t1-t2)


        right_child = Node()
        total_right = np.count_nonzero(feature_vector[:,k] > max_threshold)
        if(total_right < self.min_samples_split or depth == self.max_depth-1):
            right_child.isLeaf = True
        else:
            zeros0 = 0
            total0 = 0
            for i in range(len(feature_vector[:,k])):
                val = feature_vector[i,k]
                if(val<max_threshold):
                    continue
                if(output_vector[i] == 0):
                    zeros0 += 1
                total0 += 1
            ones0 = total0 - zeros0
            t1 = (ones0/total0)**2#Check correct power?
            t2 = (zeros0/total0)**2

            feature_cap = np.zeros((total0,len(feature_vector[0])))
            output_cap = np.zeros(total0)
            m = 0
            for i in range(len(feature_vector[:,k])):
                val = feature_vector[i,k]
                if(val<max_threshold):
                    continue
                feature_cap[m] = feature_vector[i]
                output_cap[m] = output_vector[i]
                m += 1
            
            right_child = self.ID3(depth+1,feature_cap,output_cap,1-t1-t2)
        root.children[0] = left_child
        root.children[1] = right_child
        return root

'''IG'''
train_path = "data/train"
feature_vector, output_vector = take_input(train_path,2000)
validation_path = "data/validation"
feature_val, output_val = take_input(validation_path,400)


begin = time.time()
DT = DecisionTree(10,7,feature_vector,output_vector)
initial_gini = 1#Check
root = DT.ID3(0,feature_vector,output_vector,initial_gini)
bool_vector = np.ones(2000)
DT.leafval(root,feature_vector,output_vector,bool_vector)
end = time.time()
print(f"Time taken is {end - begin} seconds")

predicted_val = predict(root,feature_val,400)
predicted_vector = predict(root, feature_vector,2000)

confusion_matrix = metrics.confusion_matrix(output_vector, predicted_vector,labels=[0,1])
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix,display_labels=[0,1])
cm_display.plot()
plt.show()

accuracy = (confusion_matrix[0][0] + confusion_matrix[1][1])/len(output_vector)
print(accuracy)
print((confusion_matrix[0][0]/(confusion_matrix[0][0] + confusion_matrix[1][0])))
print((confusion_matrix[1][1]/(confusion_matrix[1][1] + confusion_matrix[0][1])))
print((confusion_matrix[0][0]/(confusion_matrix[0][0] + confusion_matrix[0][1])))
print((confusion_matrix[1][1]/(confusion_matrix[1][1] + confusion_matrix[1][0])))

confusion_matrix = metrics.confusion_matrix(output_val, predicted_val,labels=[0,1])
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix,display_labels=[0,1])
cm_display.plot()
plt.show()

accuracy = (confusion_matrix[0][0] + confusion_matrix[1][1])/len(output_val)
print(accuracy)
print((confusion_matrix[0][0]/(confusion_matrix[0][0] + confusion_matrix[1][0])))
print((confusion_matrix[1][1]/(confusion_matrix[1][1] + confusion_matrix[0][1])))
print((confusion_matrix[0][0]/(confusion_matrix[0][0] + confusion_matrix[0][1])))
print((confusion_matrix[1][1]/(confusion_matrix[1][1] + confusion_matrix[1][0])))