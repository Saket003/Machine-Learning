from TakeInput import take_input_multiclass_2, take_input_test, csvoutput
from sklearn import tree
from sklearn.feature_selection import SelectKBest, f_classif, chi2

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--train_path")
parser.add_argument("--test_path")
parser.add_argument("--out_path")
args = vars(parser.parse_args())

train_path = args["train_path"]
test_path = args["test_path"]
out_path = args["out_path"]

full_feature_vector, output_vector = take_input_multiclass_2(train_path)
full_test_vector,test_lables = take_input_test(test_path)

best_10 = SelectKBest(score_func=f_classif, k=10) #alt = chi2
feature_vector = best_10.fit_transform(full_feature_vector,output_vector)
test_vector = best_10.transform(full_test_vector)

clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=5,min_samples_split=4,random_state=0) 
clf.fit(feature_vector,output_vector)

predicted_vector = clf.predict(test_vector)

file_name = out_path
csvoutput(predicted_vector,test_lables,file_name)