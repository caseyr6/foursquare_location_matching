import numpy as np
import pandas as pd
import random
import warnings
import datetime
import joblib
import xgboost as xgb
from pathlib import Path
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# import custom modules
import clean_data, generate_pairs, build_features, evaluate_model

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)



t0 = datetime.datetime.now()
print(f'script started @ {t0}')
print()



parent_path = str(Path(__file__).parent).replace('src','')
model_dir = 'models'
model_name = 'xgb_classifier_full_train.sav'
model_path = os.path.join(parent_path, model_dir, model_name)



#########################################
# read the raw training / validation data
#########################################

train_df = pd.read_csv(r'C:\Users\caseyrya\Dropbox\foursquare_location_matching_data\data\model_training\train_raw.csv')
val_df = pd.read_csv(r'C:\Users\caseyrya\Dropbox\foursquare_location_matching_data\data\model_training\val_raw.csv')

print(f'data read - time taken so far: {datetime.datetime.now() - t0}')
print()



##########################################
# clean the training / validation data set
##########################################

train_df = clean_data.clean_data(train_df)
val_df = clean_data.clean_data(val_df)

print(f'data cleaned - time taken so far: {datetime.datetime.now() - t0}')
print()



###############################
# generate the pairs dataframes
###############################

train_df = generate_pairs.generate_pairs(train_df)
val_df = generate_pairs.generate_pairs(val_df)

print(f'pairs generated - time taken so far: {datetime.datetime.now() - t0}')
print()



###########################
# build modelling data sets
###########################

train_df = build_features.build_features(train_df)
val_df = build_features.build_features(val_df)

train_df.to_csv(r'C:\Users\caseyrya\Dropbox\foursquare_location_matching_data\data\model_training\train_features.csv', index=False)
val_df.to_csv(r'C:\Users\caseyrya\Dropbox\foursquare_location_matching_data\data\model_training\val_features.csv', index=False)

# added in to reduce run time (build_features takes a long time - for training set especially)
# train_df = pd.read_csv(r'C:\Users\caseyrya\Dropbox\foursquare_location_matching_data\data\model_training\train_features.csv')
# val_df = pd.read_csv(r'C:\Users\caseyrya\Dropbox\foursquare_location_matching_data\data\model_training\val_features.csv')


id_pairs = val_df[['id_1','id_2','match']]

X_train = train_df.drop(['match','id_1','id_2','full_address_1_lang','full_address_2_lang'], axis = 1).copy()
X_val = val_df.drop(['match','id_1','id_2','full_address_1_lang','full_address_2_lang'], axis = 1).copy()
y_train = train_df[['match']].copy()
y_val = val_df[['match']].copy()

print(f'features built - time taken so far: {datetime.datetime.now() - t0}')
print()



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

print(f'model defined - time taken so far: {datetime.datetime.now() - t0}')
print()



######################
# train the classifier
######################

# train model
clf.fit(X_train,
        y_train,
        eval_metric = ['logloss'],
        early_stopping_rounds = 20,
        eval_set = [(X_train, y_train),
                    (X_val, y_val)],
        verbose = True)

print(f'model trained - time taken so far: {datetime.datetime.now() - t0}')
print()



#########################
# evaluate the classifier
#########################

model_acc_train, model_prec_train, model_rec_train, model_f1_train, model_auc_pr_train, model_roc_auc_train, model_comp_score_train = evaluate_model.evaluate_model(train_df, clf)
model_acc_val, model_prec_val, model_rec_val, model_f1_val, model_auc_pr_val, model_roc_auc_val, model_comp_score_val = evaluate_model.evaluate_model(val_df, clf)

print(f'model accuracy train / val: {model_acc_train} / {model_acc_val}')
print(f'model precision train / val: {model_prec_train} / {model_prec_val}')
print(f'model recall train / val: {model_rec_train} / {model_rec_val}')
print(f'model f1-score train / val: {model_f1_train} / {model_f1_val}')
print(f'model AUC PR train / val: {model_auc_pr_train} / {model_auc_pr_val}')
print(f'model AUC ROC train / val: {model_roc_auc_train} / {model_roc_auc_val}')
print(f'model competition score train / val: {model_comp_score_train} / {model_comp_score_val}')
print()

print(f'model evaluated - time taken so far: {datetime.datetime.now() - t0}')
print()



#############################
# save the trained classifier
#############################

joblib.dump(clf, Path(model_path))

print(f'model saved - total time taken: {datetime.datetime.now() - t0}')
print()