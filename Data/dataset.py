from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from macros import *
from typing import List
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from tokenizer import vocab_iterator

def create_dataset(dataset_dir):
    dataset=pd.read_csv(dataset_dir,encoding="utf-8")
    dataset=dataset[:10]
    graphemes=list(dataset["grapheme"])
    phonemes=list(dataset["phoneme"])
    all_data=[]
    for i in range(len(graphemes)):
        all_data.append((graphemes[i],phonemes[i]))
    train_dataset,test_dataset=train_test_split(all_data,test_size=0.2,random_state=42,shuffle=True)
    return train_dataset,test_dataset,all_data

def sequential_transforms(*transforms):
    def func(text_input):
        for transform in transforms:
            text_input=transform(text_input)
        return text_input
    return func

# add BOS-EOS 
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

# collate data samples into batch tesors

def collate_fn(batch):
    s_batch, t_batch = [], []
    for s_sample, t_sample in batch:
        s_batch.append(text_transform[input](s_sample.rstrip("\n")))
        t_batch.append(text_transform[output](t_sample.rstrip("\n")))
    s_batch = pad_sequence(s_batch, padding_value=PAD_IDX)
    t_batch = pad_sequence(t_batch, padding_value=PAD_IDX)
    return s_batch, t_batch

def prepare_dataset(data_dir):
    train_data,test_data,all_data=create_dataset(data_dir)
    source_vocab_Size,target_vocab_Size,vocab_transform=vocab_iterator(all_data)
    for ln in [input,output]: 
        text_transform[ln] = sequential_transforms(token_transform,vocab_transform,tensor_transform)
    return train_data,test_data,all_data,source_vocab_Size,target_vocab_Size
