import utils
import etl
import models_partb
import models_partc
import cross
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import *
from sklearn.cross_validation import *
from xgboost import XGBClassifier
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression

#Note: You can reuse code that you wrote in etl.py and models.py and cross.py over here. It might help.
# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

'''
You may generate your own features over here.
Note that for the test data, all events are already filtered such that they fall in the observation window of their respective patients.
Thus, if you were to generate features similar to those you constructed in code/etl.py for the test data, all you have to do is aggregate events for each patient.
IMPORTANT: Store your test data features in a file called "test_features.txt" where each line has the
patient_id followed by a space and the corresponding feature in sparse format.
Eg of a line:
60 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514
Here, 60 is the patient id and 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514 is the feature for the patient with id 60.

Save the file as "test_features.txt" and save it inside the folder deliverables

input:
output: X_train,Y_train,X_test
'''
def my_features():
	#TODO: complete this
	train_path = '../data/train/'
	test_path = '../data/test/'
	train_events = pd.read_csv(train_path + 'events.csv')
	train_mortality = pd.read_csv(train_path + 'mortality_events.csv')
	train_feature_map = pd.read_csv(train_path + 'event_feature_map.csv')
	
	test_events = pd.read_csv(test_path + 'events.csv')
	test_feature_map = pd.read_csv(test_path + 'event_feature_map.csv')
	
	patient_features, mortality = etl.create_features(train_events, train_mortality, train_feature_map)
	etl.save_svmlight(patient_features, mortality, '../others/features_svmlight.train', '../others/features.train')
	X_train, Y_train = utils.get_data_from_svmlight("../others/features_svmlight.train")
	
	deliverables_path = '../others/'

	aggregated_events = etl.aggregate_events(test_events[['patient_id','event_id','value']], train_mortality, test_feature_map,deliverables_path)
	merged=pd.merge(test_events,train_mortality,on='patient_id', suffixes=['_x','_y'],how='left')
	merged.fillna(0,inplace=True)
	test_patient_features=aggregated_events.groupby('patient_id')[['feature_id','feature_value']].apply(lambda x: [tuple(x) for x in x.values]).to_dict()
	test_mortality=merged.groupby('patient_id')['label'].apply(lambda x: x.unique()[0]).to_dict()
	etl.save_svmlight(test_patient_features, test_mortality, '../others/features_svmlight.test', '../others/features.test')

	
	deliverable1 = open('../deliverables/test_features.txt', 'wb')

	sorted_keys=sorted(test_patient_features.keys())
	d1=''
	for i in sorted_keys:
		deliverable1.write(str(int(i)))
		others=sorted(test_patient_features[i])
		for j in others:
			deliverable1.write(' '+str(int(j[0]))+':'+'%.6f' % (j[1]))
		deliverable1.write(' \n');
		
	X_test, Y_test = utils.get_data_from_svmlight( '../others/features_svmlight.test')
	
	return X_train,Y_train,X_test


'''
You can use any model you wish.

input: X_train, Y_train, X_test
output: Y_pred
'''
def my_classifier_predictions(X_train,Y_train,X_test):
	#TODO: complete this
	
	# base model
	logreg=LogisticRegression(C=1.0).fit(X_train,Y_train)
	
	#Cross Validation
	train_x,test_x,train_y,test_y = cross_validation.train_test_split(X_train,Y_train,test_size=0.3,random_state=0)
	## Random Forest Model
	# #grid search to find best parameter for random forest
	# rfc=RandomForestClassifier(n_jobs=-1,max_features='sqrt',n_estimators=50)
	# param_grid = {'n_estimators': [200, 300, 500, 700], 'max_features': ['auto','sqrt','log2']}
	# CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, n_jobs=-1,
	# 					   cv=10, verbose=20, scoring='roc_auc')
	# CV_rfc.fit(X_train, Y_train)
	# # print CV_rfc.best_params_
	# # print CV_rfc.best_score_
	# model=RandomForestClassifier(n_jobs=-1,max_features='log2',n_estimators=500).fit(X_train,Y_train)
	
	##XGB model
	# params = {
	# 'max_depth': [2,3,4],
	# 'n_estimators':[100,200,300],
	# 'seed':[0,550,10,200]
	# }
	# clf=XGBClassifier(learning_rate=0.1,max_depth=3,n_estimators=100, silent=True, objective= 'binary:logistic', nthread=-1, gamma=0, min_child_weight=1,
	# 				  max_delta_step=0,subsample=0.8, colsample_bytree=0.85,seed=2017)
	# CV_clf = GridSearchCV(estimator=clf, param_grid=params,n_jobs=-1,cv=10,verbose=20,scoring='roc_auc')
	# CV_clf.fit(X_train,Y_train)
	# model=XGBClassifier(learning_rate=0.02,max_depth=2,n_estimators=200, silent=True, objective= 'binary:logistic', nthread=-1, gamma=0, min_child_weight=1,
	# 				  max_delta_step=0,subsample=0.8, colsample_bytree=0.85,seed=550).fit(X_train,Y_train)
	# 
	# print CV_clf.best_params_
	# print CV_clf.best_score_
	
	
	# etc=ExtraTreesClassifier(n_jobs=-1,max_features='sqrt',n_estimators=50)
	# param_grid2 = {'n_estimators': [400, 500, 600,700,800], 'max_features': ['auto','sqrt','log2']}
	# CV_etc = GridSearchCV(estimator=etc, param_grid=param_grid2, n_jobs=-1,
	# 					   cv=10, verbose=20, scoring='roc_auc')
	# CV_etc.fit(X_train, Y_train)
	# print CV_etc.best_params_
	# print CV_etc.best_score_
	model=ExtraTreesClassifier(n_jobs=-1,max_features='auto',n_estimators=600).fit(X_train,Y_train)
	
	Y_pred=map(int,model.predict(X_test))
	#Y_pred=model.predict_proba(X_test)[:,1]
	return Y_pred



def main():
	X_train, Y_train, X_test = my_features()
	#print X_train,X_test
	Y_pred = my_classifier_predictions(X_train,Y_train,X_test)
	utils.generate_submission("../deliverables/test_features.txt",Y_pred)
	#The above function will generate a csv file of (patient_id,predicted label) and will be saved as "my_predictions.csv" in the deliverables folder.
	
	X_traintest,Y_traintest= utils.get_data_from_svmlight("../data/features_svmlight.validate")
	Y_trainpred=my_classifier_predictions(X_train,Y_train,X_traintest)
	auc=roc_auc_score(Y_traintest,Y_trainpred)
	#print auc
	#models_partc.display_metrics("My model",Y_trainpred,Y_traintest)


if __name__ == "__main__":
    main()

	