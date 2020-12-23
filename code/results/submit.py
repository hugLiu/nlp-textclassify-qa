import os

from constants import DEFAULT_SUBMISSION_PATH
from configs.config import Config

config = Config.yaml_config()

def predict_to_file(df, mode='csv'):
    if mode == 'csv':
        submit_to_csv(df)

def submit_to_csv(df):
    path = os.path.join(DEFAULT_SUBMISSION_PATH, config['submit_path'])
    df[config['submit_csv_columns']].to_csv(path, index=False, sep='\t', header=None)
