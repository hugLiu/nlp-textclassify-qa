"""
1. 基础库
2. 神经网络框架库
3. 神经网络应用库
4. 本地库
"""
import time
import numpy as np

from utils.helper import (
    set_seed,
    search_f1,
    format_time
)
from corpus import corpus
from configs.config import Config
from datasets.dataset import dataloader_generator
from results.result import do_result
from models.estate import EstateModel

# global config
config = Config.yaml_config()
# global corpus data
df_train, df_eval, df_test = corpus.load_data()
print('df_train.shape:', df_train.shape)
print('df_eval.shape:', df_eval.shape)
print('df_test.shape:', df_test.shape)

def train():
    # calculate the train time
    t0 = time.time()

    train_dataloader = dataloader_generator(df_train)
    eval_dataloader = dataloader_generator(df_eval, mode='eval')

    # init model
    model = EstateModel()
    # train, evaluate
    model.fit(train_dataloader, eval_dataloader)

    # eval predict
    eval_preds = model.predict(eval_dataloader)

    # test predict
    test_dataloader = dataloader_generator(df_test, mode='test')
    test_preds = model.predict(test_dataloader)

    total_time = format_time(time.time() - t0)
    print(f'train finished, took: {total_time}')

    return np.array(eval_preds), np.array(test_preds)

def find_threshold(eval_preds):
    # compute the threshold by all train label and all predictions
    # eval_preds contain all train set
    labels = np.array(df_eval['label'].values)
    threshold = search_f1(labels, eval_preds)

    return threshold

def main():
    # Set the seed value all over the place to make this reproducible.
    set_seed(config['seed'])
    # get preds by train
    eval_preds, test_preds = train()

    threshold = find_threshold(eval_preds)
    test_label = (test_preds > threshold).astype(int)
    # update label feature
    df_test['prediction'] = test_label

    # generate result
    do_result(df_test)

if __name__ == '__main__':
    main()
