from typing import List
from torchtext.vocab import build_vocab_from_iterator
from macros import special_symbols,input,output
from typing import Iterable
from macros import *
def my_tokenizer(word: str) -> List[str]:
    return list(word)

# yield token for build_vocab_from_iterator function
def yield_tokens(data_iter: Iterable, language: str):
    language_index = {input: 0,output: 1}
    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])



#Create  Vocab   
def vocab_iterator(all_data):
    for ln in [input, output]:
       vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(all_data, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)
    Source_vocab_size= len(vocab_transform[input])
    target_vocab_size = len(vocab_transform[output])
    return Source_vocab_size,target_vocab_size,vocab_transform
