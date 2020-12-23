import numpy as np

from results.submit import predict_to_file
from utils.helper import(
    map,
    mrr,
    search_mcc
)

def search_metrics(labels, predictions):
    acc = (labels == predictions).sum() / len(labels)
    map_score = map(labels, predictions)
    mrr_score = mrr(labels, predictions)
    mcc_score = search_mcc(labels, predictions)

    print(f'Accuracy is {acc}, MAP is {map_score}, MRR is {mrr_score}, MCC is {mcc_score}')

def do_result(df):
    labels, predictions = np.array(df['label'].values), np.array(df['prediction'].values)

    search_metrics(labels, predictions)
    predict_to_file(df, mode='csv')
