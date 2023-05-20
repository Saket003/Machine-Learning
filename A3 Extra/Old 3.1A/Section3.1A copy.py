from TakeInput import take_input
import time
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

#log2 everywhere


def mapper(vector): #TODO Modify for better binary split
    thresholds = np.zeros(len(vector[0]))
    for j in range(len(vector[0])):
        column = vector[:,j]    #Approximating threshold using distributions - normal
        upper0 = column[0:1500]
        lower1 = column[1500:2000]
        mean0 = np.mean(upper0)
        mean1 = np.mean(lower1)
        if(mean0 > mean1):
            C = (np.quantile(upper0,0.25)+np.quantile(lower1,0.75))/2
        else:
            C = (np.quantile(upper0,0.75)+np.quantile(lower1,0.25))/2
        thresholds[j] = C
    return thresholds

def mappee(vector, thresholds):
    n_features = len(vector[0])

    for i in range(n_features):
        vector[:,i] = (vector[:,i] > thresholds[i]).astype(int)

    return vector

def predict(root,feature_val):
    predicted_val = np.zeros(400)   #Check better?
    i = 0
    safe_root = root
    for features in feature_val:
        root = safe_root
        while(True):
            if(root.isLeaf == True):
                predicted_val[i] = root.leafvalue
                i += 1
                break
            root = root.children[int(features[root.para_index])]
    return predicted_val

class Node:
    def __init__(self,para_index=None):
        self.para_index = para_index
        self.isLeaf = False
        self.leafvalue = None
        self.children = [None, None]


class DecisionTree:
    def __init__(self,max_depth,min_samples_split):
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def ID3(self,depth,unused_indices,feature_vector, output_vector, parent_entropy):
        root = Node()
        
        k = 0
        GR = np.zeros(len(unused_indices))
        m = 0
        for i in unused_indices:
            entropy = [None, None]
            count_group = [None, None]
            for j in range(2):
                ones = 0
                total = np.count_nonzero(feature_vector[:,i] == j)
                count_group[j] = total
                for w in range(len(output_vector)):
                    if(feature_vector[w,i]==j and output_vector[w]==1):
                        ones += 1
                zeros = total - ones
                if(zeros == 0 or ones == 0):
                    entropy[j] = 0
                else:
                    entropy[j] = (ones/total)*np.log2(total/ones) + (zeros/total)*np.log2(total/zeros)
            avg_entropy = np.average(entropy)
            gain = parent_entropy - avg_entropy

            sum = 0
            c_t = np.sum(count_group)
            for c in count_group:
                if (c==0):
                    continue
                sum = sum + (c/c_t)*np.log2(c_t/c)
            split_information = sum

            #if(split_information == 0):
                #GR[m] = -1
            #else:
            GR[m] = gain/split_information
            m += 1
        
        temp = np.argmax(GR)
        if(type(temp)==np.ndarray):
            max_indice = temp[0]
        else:
            max_indice = temp
        k = unused_indices[max_indice]
        root.para_index = k
        unused_indices.remove(k)

        for i in range(2):
            child = Node()
            total = np.count_nonzero(feature_vector[:,k] == i)
            if(total < self.min_samples_split or depth == self.max_depth-1):
                child.isLeaf = True
                ones = 0
                for j in range(len(output_vector)):
                    if(feature_vector[j,k]==i and output_vector[j]==1):
                        ones += 1
                zeros = total - ones
                child.isLeaf = True
                child.leafvalue = 1 if (3*ones>zeros) else 0    #Check 3* change
            else:
                entropy = 0
                ones = 0
                total = np.count_nonzero(feature_vector[:,k] == i)
                for w in range(len(output_vector)):
                    if(feature_vector[w,k]==i and output_vector[w]==1):
                        ones += 1
                zeros = total - ones
                if(zeros == 0 or ones == 0):
                    entropy = 0
                else:
                    entropy = (ones/total)*np.log2(total/ones) + (zeros/total)*np.log2(total/zeros)
                child = self.ID3(depth+1,unused_indices,feature_vector,output_vector,entropy)
            root.children[i] = child
        return root

'''IG'''

train_path = "data/train"
feature_vector, output_vector = take_input(train_path,2000)
validation_path = "data/validation"
feature_val, output_val = take_input(validation_path,400)
thresholds = mapper(feature_vector)
feature_vector = mappee(feature_vector,thresholds)
feature_val = mappee(feature_val,thresholds)    #Only 0s and 1s
indices = [i for i in range(3072)]

begin = time.time()
DT = DecisionTree(4,7)
initial_entropy =  (3/4)*np.log2(4/3) + (1/4)*np.log2(4)
root = DT.ID3(0,indices,feature_vector,output_vector,initial_entropy)
end = time.time()
print(f"Time taken is {end - begin} seconds")

predicted_val = predict(root,feature_val)

#Confusion for both train and val
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