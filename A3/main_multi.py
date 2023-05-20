from TakeInput import take_input_multiclass, take_input_test, csvoutput
from xgboost import XGBClassifier

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--train_path")
parser.add_argument("--test_path")
parser.add_argument("--out_path")
args = vars(parser.parse_args())

train_path = args["train_path"]
test_path = args["test_path"]
out_path = args["out_path"]

feature_vector, output_vector = take_input_multiclass(train_path,2000)
test_vector,test_lables = take_input_test(test_path,18)

gs = XGBClassifier(random_state = 0,max_depth=10,n_estimators=50,subsample=0.6)
gs.fit(feature_vector,output_vector)
predicted_vector = gs.predict(test_vector)

file_name = out_path
csvoutput(predicted_vector,test_lables,file_name)