from TakeInput import take_input_multiclass_2, take_input_test, csvoutput
from sklearn.ensemble import RandomForestClassifier

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--train_path")
parser.add_argument("--test_path")
parser.add_argument("--out_path")
args = vars(parser.parse_args())

train_path = args["train_path"]
test_path = args["test_path"]
out_path = args["out_path"]

feature_vector, output_vector = take_input_multiclass_2(train_path)
test_vector,test_lables = take_input_test(test_path)

clf = RandomForestClassifier(random_state=0,criterion='entropy',max_depth=10,min_samples_split=10,n_estimators=100) 
clf.fit(feature_vector,output_vector)
predicted_vector = clf.predict(test_vector)

file_name = out_path
csvoutput(predicted_vector,test_lables,file_name)