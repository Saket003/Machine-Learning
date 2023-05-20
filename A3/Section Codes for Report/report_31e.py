from TakeInput import take_input
import time
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from confusion import confusion

#Input Set
train_path = "data/train"
feature_vector, output_vector = take_input(train_path,2000)
validation_path = "data/validation"
feature_val, output_val = take_input(validation_path,400)

'''Default'''
begin = time.time()
clf = RandomForestClassifier(random_state=0) 
clf.fit(feature_vector,output_vector)
end = time.time()
print(f"Time taken is {end - begin} seconds")

confusion(feature_vector,output_vector,clf)
confusion(feature_val,output_val,clf)

'''#GS Step
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

confusion(feature_vector,output_vector,gs)
confusion(feature_val,output_val,gs)
'''

'''Best Parameters'''
begin = time.time()
clf = RandomForestClassifier(random_state=0,criterion = 'entropy', max_depth = None, min_samples_split = 7, n_estimators = 150) 
clf.fit(feature_vector,output_vector)
end = time.time()
print(f"Time taken is {end - begin} seconds")

confusion(feature_vector,output_vector,clf)
confusion(feature_val,output_val,clf)