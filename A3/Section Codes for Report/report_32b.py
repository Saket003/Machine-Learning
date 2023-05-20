from TakeInput import take_input_multiclass
from sklearn import tree
import time
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.model_selection import GridSearchCV
from confusion import accuracy

#Input Set
train_path = "data/train"
full_feature_vector, output_vector = take_input_multiclass(train_path,2000)
validation_path = "data/validation"
full_feature_val, output_val = take_input_multiclass(validation_path,400)

'''SelectKBest'''
#Feature selection
best_10 = SelectKBest(score_func=f_classif, k=10) #alt = chi2
feature_vector = best_10.fit_transform(full_feature_vector,output_vector)
feature_val = best_10.transform(full_feature_val)

begin = time.time()
clf = tree.DecisionTreeClassifier(criterion='gini',max_depth=10,min_samples_split=7,random_state=0) 
clf.fit(feature_vector,output_vector)
end = time.time()
print(f"Time taken is {end - begin} seconds")
tree.plot_tree(clf)

accuracy(feature_vector,output_vector,clf)
accuracy(feature_val,output_val,clf)

'''#Grid-Search
begin = time.time()
params = [{'criterion':['gini','entropy'],
           'max_depth':[None,5,7,10,15],
           'min_samples_split':[2,4,7,9]}]
gs = GridSearchCV(tree.DecisionTreeClassifier(random_state=0),param_grid=params,scoring='accuracy',cv=5) 
gs.fit(feature_vector,output_vector)
print(gs.best_params_)
end = time.time()
print(f"Time taken is {end - begin} seconds")

accuracy(feature_vector,output_vector,gs)
accuracy(feature_val,output_val,gs)
'''

#Best parameters
begin = time.time()
clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=5,min_samples_split=4,random_state=0) 
clf.fit(feature_vector,output_vector)
end = time.time()
print(f"Time taken is {end - begin} seconds")

accuracy(feature_vector,output_vector,clf)
accuracy(feature_val,output_val,clf)