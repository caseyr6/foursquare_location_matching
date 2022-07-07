import numpy as np
import pandas as pd
import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, auc, roc_auc_score, roc_curve

def jaccard_similarity(
    list1,
    list2
):
    '''
    Function takes the two lists as input.
    
    Returns the jaccard similarity of the two lists.
    '''
    
    s1 = set(list1)
    s2 = set(list2)
    
    return float(len(s1.intersection(s2)) / len(s1.union(s2))) 


def evaluate_model(
    data,
    model,
    pos_threshold = 0.3
):
    '''
    Function takes a data set and a trained model as inputs. User can also change the pos_threshold (threshold above which a sample is classes as positive) if needed.
    
    Performs operations required to generate all of the model evaluation metrics, both standard and competition specific.
    
    Returns the 7 calculated metrics.
    '''
    
    id_pairs_true = data[['id_1','id_2','match']].copy()
    X = data.drop(['match','id_1','id_2','full_address_1_lang','full_address_2_lang'], axis = 1).copy()
    
    preds_df = id_pairs_true.copy()
    preds_df['prob_false'] = [i[0] for i in model.predict_proba(X)]
    preds_df['prob_true'] = [i[1] for i in model.predict_proba(X)]

    preds_df['model_pred'] = np.where(preds_df['prob_true']>=pos_threshold, True, False)
    
    # define required figure sets
    actuals = preds_df['match'].copy()
    preds = preds_df['model_pred'].copy()
    pos_pred_probs = preds_df['prob_true'].copy()
    
    # calculate classic metrics
    model_acc = round(accuracy_score(actuals, preds), 4)
    model_prec = round(precision_score(actuals, preds), 4)
    model_rec = round(recall_score(actuals, preds), 4)
    model_f1 = round(f1_score(actuals, preds), 4)
    
    try:
        prec, rec, _ = precision_recall_curve(actuals, pos_pred_probs)
        model_auc_pr = round(auc(rec, prec), 4)
    except:
        model_auc_pr = np.nan
    
    try:
        fpr, tpr, _ = roc_curve(actuals, pos_pred_probs)
        model_auc_roc = round(roc_auc_score(actuals, pos_pred_probs), )
    except:
        model_auc_roc = np.nan
    
    # calculate competition specific metric (mean jaccard similarity score)
    unique_ids = []
    actual_matches = []
    predicted_matches = []
    
    count = 0

    for unique_id in preds_df.id_1.drop_duplicates().tolist():

        id_actual_matches = []
        id_predicted_matches = []

        id_df = preds_df[preds_df['id_1']==unique_id].copy()

        for row in range(0, len(id_df)):

            if id_df['match'][row:row+1].values[0] == True:
                id_actual_matches.append(id_df['id_2'][row:row+1].values[0])
            else:
                pass

            if id_df['model_pred'][row:row+1].values[0] == True:
                id_predicted_matches.append(id_df['id_2'][row:row+1].values[0])
            else:
                pass

        id_actual_matches.append(unique_id)
        id_predicted_matches.append(unique_id)

        unique_ids.append(unique_id)
        actual_matches.append(id_actual_matches)
        predicted_matches.append(id_predicted_matches)
        
        count +=1

        if count % 10000 == 0:
            print('{} / {} complete. time: {}'.format(count, len(preds_df.id_1.drop_duplicates().tolist()), datetime.datetime.now()))


    comp_metric_df = pd.DataFrame({'id': unique_ids,
                                   'actual_matches': actual_matches,
                                   'predicted_matches': predicted_matches})
    
    comp_metric_df['jaccard_score'] = comp_metric_df.apply(lambda x: round(jaccard_similarity(x['predicted_matches'], x['actual_matches']),4), axis=1)
    
    model_comp_score = round(comp_metric_df.jaccard_score.mean(), 4)
    
    return model_acc, model_prec, model_rec, model_f1, model_auc_pr, model_auc_roc, model_comp_score