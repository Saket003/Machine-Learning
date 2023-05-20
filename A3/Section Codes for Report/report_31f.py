from TakeInput import take_input
import time
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from confusion import confusion

#Input Set
train_path = "data/train"
feature_vector, output_vector = take_input(train_path,2000)
validation_path = "data/validation"
feature_val, output_val = take_input(validation_path,400)

'''' #TODO - Code for Grad Boosted GS is correct, but unable to run in sufficient time
#Gradient boosted Grid-Search
begin = time.time()
params = [{'n_estimators':[20,30,40,50],
           'subsample':[0.2,0.3,0.4,0.5,0.6],
           'max_depth':[5,6,7,8,9,10]}]
gs = GridSearchCV(GradientBoostingClassifier(random_state=0),param_grid=params,scoring='accuracy',cv=5,n_jobs=-1) 
#Added n_jobs = -1 to improve performance
gs.fit(feature_vector,output_vector)
print(gs.best_params_)
end = time.time()
print(f"Time taken is {end - begin} seconds")

confusion(feature_vector,output_vector,gs)
confusion(feature_val,output_val,gs)
'''

'''Gradient boosted Default'''
begin = time.time()
gs = GradientBoostingClassifier(random_state=0)
gs.fit(feature_vector,output_vector)
end = time.time()
print(f"Time taken is {end - begin} seconds")

confusion(feature_vector,output_vector,gs)
confusion(feature_val,output_val,gs)

'''XGBoost Default'''
begin = time.time()
gs = XGBClassifier(random_state = 0)
gs.fit(feature_vector,output_vector)
end = time.time()
print(f"Time taken is {end - begin} seconds")

confusion(feature_vector,output_vector,gs)
confusion(feature_val,output_val,gs)


'''#XGBoost Grid-Search
begin = time.time()
params = [{'n_estimators':[20,30,40,50],
           'subsample':[0.2,0.3,0.4,0.5,0.6],
           'max_depth':[5,6,7,8,9,10]}]
gs = GridSearchCV(XGBClassifier(random_state = 0),param_grid=params,scoring='accuracy',cv=5) 
gs.fit(feature_vector,output_vector)
print(gs.best_params_)
end = time.time()
print(f"Time taken is {end - begin} seconds")

confusion(feature_vector,output_vector,gs)
confusion(feature_val,output_val,gs)
'''

#XGBoost Best Parameters
begin = time.time()
gs = XGBClassifier(random_state = 0,max_depth = 6,n_estimators = 40,subsample = 0.6)
gs.fit(feature_vector,output_vector)
end = time.time()
print(f"Time taken is {end - begin} seconds")

confusion(feature_vector,output_vector,gs)
confusion(feature_val,output_val,gs)