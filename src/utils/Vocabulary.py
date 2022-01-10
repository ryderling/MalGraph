import json
import os
from collections import Counter
from tqdm import tqdm


class Vocab:
    def __init__(self, freq_file: str, max_vocab_size: int, min_freq: int = 1, unk_token: str = '<unk>', pad_token: str = '<pad>', special_tokens: list = None):
        
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.special_tokens = special_tokens
        
        assert os.path.exists(freq_file), "The file of {} is not exist".format(freq_file)
        freq_counter = self.load_freq_counter_from_file(file_path=freq_file, min_freq=self.min_freq)
        
        self.token_2_index, self.index_2_token = self.create_vocabulary(freq_counter=freq_counter)
        
        self.unk_idx = None if self.unk_token is None else self.token_2_index[self.unk_token]
        self.pad_idx = None if self.pad_token is None else self.token_2_index[self.pad_token]
    
    def __len__(self):
        return len(self.index_2_token)
    
    def __getitem__(self, item: str):
        assert isinstance(item, str)
        if item in self.token_2_index.keys():
            return self.token_2_index[item]
        else:
            if self.unk_token is not None:
                return self.token_2_index[self.unk_token]
            else:
                raise KeyError("{} is not in the vocabulary, and self.unk_token is None".format(item))
    
    def create_vocabulary(self, freq_counter: Counter):
        
        token_2_index = {}  # dict
        index_2_token = []  # list
        
        if self.unk_token is not None:
            index_2_token.append(self.unk_token)
        if self.pad_token is not None:
            index_2_token.append(self.pad_token)
        if self.special_tokens is not None:
            for token in self.special_tokens:
                index_2_token.append(token)
        
        for f_name, count in tqdm(freq_counter.most_common(self.max_vocab_size), desc="creating vocab ... "):
            if f_name in index_2_token:
                print("trying to add {} to the vocabulary, but it already exists !!!".format(f_name))
                continue
            else:
                index_2_token.append(f_name)
        
        for index, token in enumerate(index_2_token):  # reverse
            token_2_index.update({token: index})
        
        return token_2_index, index_2_token
    
    @staticmethod
    def load_freq_counter_from_file(file_path: str, min_freq: int):
        freq_dict = {}
        with open(file_path, 'r') as f:
            for line in tqdm(f, desc="Load frequency list from the file of {} ... ".format(file_path)):
                line = json.loads(line)
                f_name = line["f_name"]
                count = int(line["count"])
                
                assert f_name not in freq_dict, "trying to add {} to the vocabulary, but it already exists !!!"
                if count < min_freq:
                    print(line, "break")
                    break
                
                freq_dict[f_name] = count
        return Counter(freq_dict)


if __name__ == '__main__':
    max_vocab_size = 1000
    vocab = Vocab(freq_file="../../data/processed_dataset/train_external_function_name_vocab.jsonl", max_vocab_size=max_vocab_size)
    print(len(vocab.token_2_index), vocab.token_2_index)
    print(len(vocab.index_2_token), vocab.index_2_token)
    print(vocab.unk_token, vocab.unk_idx)
    print(vocab.pad_token, vocab.pad_idx)
    print(vocab['queryperformancecounter'])
    print(vocab['EmptyClipboard'])
    print(vocab[str.lower('EmptyClipboard')])
    print(vocab['X_Y_Z'])