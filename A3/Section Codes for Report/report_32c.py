import matplotlib.pyplot as plt
from TakeInput import take_input_multiclass
from sklearn import tree
from confusion import accuracy

train_path = "data/train"
feature_vector, output_vector = take_input_multiclass(train_path,2000)
validation_path = "data/validation"
feature_val, output_val = take_input_multiclass(validation_path,400)

clf = tree.DecisionTreeClassifier(random_state=0) #Consistent plot
path = clf.cost_complexity_pruning_path(feature_vector,output_vector)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

clfs = []
for ccp_alpha in ccp_alphas:
    clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(feature_vector, output_vector)
    clfs.append(clf)

test_scores = [clf.score(feature_val, output_val) for clf in clfs]
best_index_temp = test_scores.index(max(test_scores))
print(best_index_temp)
print("Best alpha:",ccp_alphas[best_index_temp])

predicted_val = clfs[best_index_temp].predict(feature_val)

accuracy(feature_vector,output_vector,clfs[best_index_temp])
accuracy(feature_val,output_val,clfs[best_index_temp])