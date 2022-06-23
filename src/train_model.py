import numpy as np
import pandas as pd
import random
import warnings
import joblib
import xgboost as xgb
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# import custom modules
import clean_data
import generate_pairs



pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)



#########################################
# read the raw training / validation data
#########################################

train_df = pd.read_csv(r'C:\Users\caseyrya\Dropbox\foursquare_location_matching_data\data\model_training\train_raw.csv')
val_df = pd.read_csv(r'C:\Users\caseyrya\Dropbox\foursquare_location_matching_data\data\model_training\val_raw.csv')



##########################################
# clean the training / validation data set
##########################################

train_df = clean_data.clean_data(train_df)
val_df = clean_data.clean_data(val_df)



###############################
# generate the pairs dataframes
###############################

train_df = generate_pairs.generate_pairs(train_df)
val_df = generate_pairs.generate_pairs(val_df)



###########################
# build modelling data sets
###########################
'''add code here once build_features module is completed'''



#######################
# define the classifier
#######################

# define the xgb params
params = {'objective': 'binary:logistic',
          'base_score': 0.5,
          'booster': 'gbtree',
          'colsample_bylevel': 1,
          'colsample_bynode': 1,
          'colsample_bytree': 0.8,
          'gamma': 0,
          'gpu_id': -1,
          'interaction_constraints': '',
          'importance_type': 'total_gain',
          'learning_rate': 0.01,
          'max_delta_step': 0,
          'max_depth': 30,
          'min_child_weight': 3,
          'monotone_constraints': '()',
          'n_jobs': 56,
          'num_parallel_tree': 1,
          'predictor': 'auto',
          'random_state': 42,
          'reg_alpha': 0.01,
          'reg_lambda': 0,
          'scale_pos_weight': 1,
          'subsample': 0.6,
          'tree_method': 'exact',
          'validate_parameters': 1,
          'verbosity': None,
          'eval_metric': 'logloss', # maybe 'aucpr' or 'binary:logistic'
          'n_estimators': 5000} # with early stopping

# define the classifier (xgboost for now)
clf = xgb.XGBClassifier(**params)



######################
# train the classifier
######################

# train model
clf.fit(X_train,
        y_train,
        eval_metric = ['logloss'], #'aucpr'],
        early_stopping_rounds = 20,
        eval_set = [(X_train, y_train),
                    (X_val, y_val)],
        verbose = True)



#########################
# evaluate the classifier
#########################

# store any key train/val performance metrics (can compare with test set figs to determine if model over/under trained)
    # will likely come from evaluate_model.py module (using eval set instead of test set)
        # output of this module will need to be key classification metrics, and the competition specific metric



#############################
# save the trained classifier
#############################

file_location = 'models_file_path/xgb_classifier_v1.sav'
joblib.dump(clf, file_location)






















