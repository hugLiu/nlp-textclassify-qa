import torch
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer

from configs.config import Config

config = Config.yaml_config()

class EstateDataset(Dataset):
    def __init__(self, df, mode='train'):
        super(EstateDataset, self).__init__()
        # datasource, dataframe
        self.df = df
        self.mode = mode
        self.questions = self.df['q'].values
        self.answers = self.df['a'].values

        self.bert_tokenizer = BertTokenizer.from_pretrained(config['bert_name'])
        self.max_seq_len = config['max_seq_len_train']
        if self.mode == 'test':
            self.max_seq_len = config['max_seq_len_test']

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, item):
        question = self.questions[item]
        answer = self.answers[item]

        encode_dict = self.bert_tokenizer.encode_plus(
            question,
            answer,
            add_special_tokens=True,
            truncation='longest_first',
            max_length=self.max_seq_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encode_dict['input_ids'].flatten()
        attention_mask = encode_dict['attention_mask'].flatten()
        token_type_ids = encode_dict['token_type_ids'].flatten()

        assert len(input_ids) == self.max_seq_len, \
            f'Error with input length {len(input_ids)} vs {self.max_seq_len}'
        assert len(attention_mask) == self.max_seq_len, \
            f'Error with input length {len(attention_mask)} va {self.max_seq_len}'
        assert len(token_type_ids) == self.max_seq_len, \
            f'Error with input length {len(token_type_ids)} vs {self.max_seq_len}'

        ret = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        }

        if self.mode != 'test':
            label = self.df['label'].values
            ret['label'] = torch.tensor(label[item], dtype=torch.float32)

        return ret

def dataloader_generator(df, mode='train'):
    shuffle = False
    if mode == 'train':
        shuffle = True

    dataset = EstateDataset(df, mode)

    return DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=config['batch_size']
    )
