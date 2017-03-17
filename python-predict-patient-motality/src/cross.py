import models_partc
from sklearn.cross_validation import KFold, ShuffleSplit
from numpy import mean

import utils

# USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

# USE THIS RANDOM STATE FOR ALL OF YOUR CROSS
# VALIDATION TESTS OR THE TESTS WILL NEVER PASS
RANDOM_STATE = 545510477

#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_kfold(X,Y):
	#TODO:First get the train indices and test indices for each iteration
	#Then train the classifier accordingly
	#Report the mean accuracy and mean auc of all the folds
	kf=KFold(len(Y),n_folds=5,random_state=RANDOM_STATE)
	accuracy=[]
	auc=[]
	for train_index, test_index in kf:
		X_train,X_test=X[train_index],X[test_index]
		Y_train,Y_test=Y[train_index],Y[test_index]
		Y_pred=models_partc.logistic_regression_pred(X_train, Y_train, X_test)
		accuracy.append(models_partc.classification_metrics(Y_pred, Y_test)[0])
		auc.append(models_partc.classification_metrics(Y_pred, Y_test)[1])
	return mean(accuracy), mean(auc)


#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_randomisedCV(X,Y):
	#TODO: First get the train indices and test indices for each iteration
	#Then train the classifier accordingly
	#Report the mean accuracy and mean auc of all the iterations
	rs=ShuffleSplit(len(Y), n_iter=5,test_size=0.2,random_state=RANDOM_STATE)
	accuracylist=[]
	auclist=[]
	for train_index, test_index in rs:
		X_train,X_test=X[train_index],X[test_index]
		Y_train,Y_test=Y[train_index],Y[test_index]
		Y_pred=models_partc.logistic_regression_pred(X_train, Y_train, X_test)
		accuracy, auc, precision, recall, f1score=models_partc.classification_metrics(Y_pred, Y_test)
		accuracylist.append(accuracy)
		auclist.append(auc)

	return mean(accuracylist), mean(auclist)


def main():
	X,Y = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
	print "Classifier: Logistic Regression__________"
	acc_k,auc_k = get_acc_auc_kfold(X,Y)
	print "Average Accuracy in KFold CV: "+str(acc_k)
	print "Average AUC in KFold CV: "+str(auc_k)
	acc_r,auc_r = get_acc_auc_randomisedCV(X,Y)
	print "Average Accuracy in Randomised CV: "+str(acc_r)
	print "Average AUC in Randomised CV: "+str(auc_r)

if __name__ == "__main__":
	main()

