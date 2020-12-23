import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn

from transformers import (
    BertModel,
    AdamW,
    get_linear_schedule_with_warmup
)

from utils.helper import (
    search_auc,
    search_f1,
    show_dataframe
)
from configs.config import Config
from constants import DEFAULT_MODEL_PATH

config = Config.yaml_config()

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.num_labels = config['num_labels']

        self.bert = BertModel.from_pretrained(config['bert_name'])
        self.dropout = nn.Dropout(p=config['hidden_dropout'])
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.num_labels)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None
        ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # last_hidden_state [batch_size, seq_len, hidden_size]
        pooler_output = outputs[1] # [batch_size, hidden_size]
        output = self.dropout(pooler_output) # [batch_size, hidden_size]
        output = self.classifier(output) # [batch_size, num_labels]
        return torch.sigmoid(output)

class EstateModel(object):
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f'We will use the GPU: {torch.cuda.get_device_name(0)} .')
        else:
            self.device = torch.device('cpu')
            print('No GPU available, using the CPU instead.')

        self.model = BaseModel().to(self.device)
        self.loss_fct = nn.BCELoss()
        self.optimizer = AdamW(
            params=self.model.parameters(),
            lr=config['lr'],
            eps=config['eps']
        )

    def fit(self, train_dataloader, eval_dataloader):
        stats_list, eval_auc_list = [], [0]

        for epoch in range(config['epochs']):
            stats_dict = {'epoch': epoch + 1}
            self.train(train_dataloader, stats_dict)
            self.evaluate(eval_dataloader, stats_dict)

            stats_list.append(stats_dict)
            eval_auc = stats_dict['eval_auc']
            if max(eval_auc_list) < eval_auc:
                self.save_model(stats_dict)
            eval_auc_list.append(eval_auc)

        columns = ['epoch', 'avg_train_loss', 'train_auc', 'avg_eval_loss', 'eval_auc', 'threshold']
        df_stats = pd.DataFrame(data=stats_list, columns=columns)
        # df_stats = df_stats.set_index('epoch')
        # figure stats by plt
        show_dataframe(df_stats)

    def train(self, dataloader, stats_dict):
        self.model.train()

        total_steps = len(dataloader) * config['epochs']
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        total_loss = 0.0
        predictions, labels = [], []

        train_iterator = tqdm(dataloader, desc='train')
        for step, batch in enumerate(train_iterator):
            label = batch['label'].to(self.device)
            inputs = self.input_features(batch)

            # clear gradient
            self.model.zero_grad()
            output = self.model(**inputs)

            # cross entropy loss
            loss = self.loss_fct(output.squeeze(1), label)
            loss_item = loss.item()
            if config['show_epoch_loss'] and (step + 1) % config['loss_step'] == 0:
                print(f'{step + 1}, loss: {loss_item}')
            # total loss
            total_loss += loss_item

            # perform a backward pass to calculate the gradients
            loss.backward()
            # clip the norm of the gradients to max_grad_norm
            # help prevent the 'exploding gradients' problem
            nn.utils.clip_grad_norm_(self.model.parameters(), config['max_grad_norm'])
            # compute gradient
            self.optimizer.step()
            # update learning rate
            scheduler.step()

            # to numpy
            prediction = output.detach().cpu().numpy().flatten()
            label = label.cpu().numpy().flatten()
            predictions.extend(prediction)
            labels.extend(label)

        avg_loss = total_loss / len(dataloader)
        stats_dict['avg_train_loss'] = avg_loss
        train_auc = search_auc(labels, predictions)
        stats_dict['train_auc'] = train_auc

    def evaluate(self, dataloader, stats_dict):
        self.model.eval()

        total_loss = 0.
        predictions, labels = [], []
        with torch.no_grad():
            for step, batch in enumerate(tqdm(dataloader, desc='eval')):
                label = batch['label'].to(self.device)

                inputs = self.input_features(batch)
                output = self.model(**inputs)

                loss = self.loss_fct(output.squeeze(1), label)
                loss_item = loss.item()
                if config['show_epoch_loss'] and (step + 1) % config['loss_step'] == 0:
                    print(f'{step + 1}, loss: {loss_item}')
                # total loss
                total_loss += loss_item

                prediction = output.cpu().numpy().flatten()
                label = label.cpu().numpy().flatten()
                predictions.extend(prediction)
                labels.extend(label)

        avg_loss = total_loss / len(dataloader)
        stats_dict['avg_eval_loss'] = avg_loss
        eval_auc = search_auc(labels, predictions)
        stats_dict['eval_auc'] = eval_auc

        labels, predictions = np.array(labels), np.array(predictions)
        threshold = search_f1(labels, predictions)
        stats_dict['threshold'] = threshold

    def predict(self, dataloader):
        self.load_model()
        self.model.eval()

        predictions = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='predict'):
                inputs = self.input_features(batch)
                output = self.model(**inputs)

                prediction = output.cpu().numpy().flatten()
                predictions.extend(prediction)
        return predictions

    def input_features(self, batch):
        return {
            'input_ids': batch['input_ids'].to(self.device),
            'attention_mask': batch['attention_mask'].to(self.device),
            'token_type_ids': batch['token_type_ids'].to(self.device)
        }

    def save_model(self, stats_dict):
        path = os.path.join(DEFAULT_MODEL_PATH, config['model_path'])

        model = self.model.cpu()
        optimizer = self.optimizer
        checkpoint = {
            'epoch': stats_dict['epoch'],
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'eval_auc': stats_dict['eval_auc'],
            'threshold': stats_dict['threshold']
        }

        torch.save(checkpoint, path)
        print(f'save model, epoch: {stats_dict["epoch"]}, eval_auc: {stats_dict["eval_auc"]} .')
        # recover device before train
        self.model.to(self.device)

    def load_model(self):
        torch.cuda.empty_cache()

        path = os.path.join(DEFAULT_MODEL_PATH, config['model_path'])
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.model.to(self.device)
