Codes in "Section Codes for CSV Generation" to test accuracy.
Shift a code from this directory to parent directory and then run shell. (parent directory must also contain data folder with same directory structure)

Codes in "Section Codes for Report" are purely for evidence of data and working put in report. They require the folder 'data' to be present in this repository.

Example of .sh - 
python code_31a.py --train_path="./data/train" --test_path="./data/test_sample" --out_path="./test_31a.csv"
python a3_eval.py

confusion and TakeInput are helper codes required by all codes.

Note:
For 3.1D and 3.2C, best pruned tree is being used, since no validation data is being passed.
(On the validation data provided to us).

Both main_binary and main_multi make use of scikit-learn(as was clarified in piazza).
Classifier is selected on the basis of accuracy and F1 score.