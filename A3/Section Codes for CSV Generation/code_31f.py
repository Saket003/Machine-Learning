from TakeInput import take_input_2, take_input_test, csvoutput
from xgboost import XGBClassifier

train_path = "data/train"
feature_vector, output_vector = take_input_2(train_path)
test_path = "data/test_sample"
test_vector,test_lables = take_input_test(test_path)

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--train_path")
parser.add_argument("--test_path")
parser.add_argument("--out_path")
args = vars(parser.parse_args())

train_path = args["train_path"]
test_path = args["test_path"]
out_path = args["out_path"]

gs = XGBClassifier(random_state = 0,max_depth = 6,n_estimators = 40,subsample = 0.6)
gs.fit(feature_vector,output_vector)
predicted_vector = gs.predict(test_vector)

file_name = out_path
csvoutput(predicted_vector,test_lables,file_name)