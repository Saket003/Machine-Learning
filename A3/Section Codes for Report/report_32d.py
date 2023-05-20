from TakeInput import take_input_multiclass
import time
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from confusion import accuracy

#Input Set
train_path = "data/train"
feature_vector, output_vector = take_input_multiclass(train_path,2000)
validation_path = "data/validation"
feature_val, output_val = take_input_multiclass(validation_path,400)

'''Default'''
begin = time.time()
clf = RandomForestClassifier(random_state=0) 
clf.fit(feature_vector,output_vector)
end = time.time()
print(f"Time taken is {end - begin} seconds")

accuracy(feature_vector,output_vector,clf)
accuracy(feature_val,output_val,clf)


'''#Grid-Search
begin = time.time()
params = [{'n_estimators':[80,100,150,200],
           'criterion':['gini','entropy'],
           'max_depth':[None,5,7,10],
           'min_samples_split':[5,7,10]}]
gs = GridSearchCV(RandomForestClassifier(random_state=0),param_grid=params,scoring='accuracy',cv=5) 
gs.fit(feature_vector,output_vector)
print(gs.best_params_)
end = time.time()
print(f"Time taken is {end - begin} seconds")

accuracy(feature_vector,output_vector,gs)
accuracy(feature_val,output_val,gs)
'''

'''Best parameters'''
begin = time.time()
clf = RandomForestClassifier(random_state=0,criterion='entropy',max_depth=10,min_samples_split=10,n_estimators=100) 
clf.fit(feature_vector,output_vector)
end = time.time()
print(f"Time taken is {end - begin} seconds")

accuracy(feature_vector,output_vector,clf)
accuracy(feature_val,output_val,clf)