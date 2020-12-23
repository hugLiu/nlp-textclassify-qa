import pandas as pd

from configs.config import Config

default_config = Config.yaml_config()

def load_data():
    # take special columns
    usecols = [0, 1, 5, 6]
    df_train = pd.read_csv('corpus/tsv/WikiQA-train.tsv', usecols=usecols, sep='\t')
    df_eval = pd.read_csv('corpus/tsv/WikiQA-dev.tsv', usecols=usecols, sep='\t')
    df_test = pd.read_csv('corpus/tsv/WikiQA-test.tsv', usecols=usecols, sep='\t')

    # rename columns
    columns = ['id', 'q', 'a', 'label']
    df_train.columns, df_eval.columns, df_test.columns = columns, columns, columns

    if 'corpus_size' in default_config.keys():
        corpus_size = default_config['corpus_size']
        return df_train[:corpus_size], df_eval[:corpus_size], df_test[:corpus_size]

    return df_train, df_eval, df_test
