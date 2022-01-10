import json
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from utils.Vocabulary import Vocab


def parse_json_list_2_pyg_object(jsonl_file: str, label: int, vocab: Vocab):
    index = 0
    with open(jsonl_file, "r", encoding="utf-8") as file:
        for item in tqdm(file):
            item = json.loads(item)
            item_hash = item['hash']
            
            acfg_list = []
            for one_acfg in item['acfg_list']:  # list of dict of acfg
                block_features = one_acfg['block_features']
                block_edges = one_acfg['block_edges']
                one_acfg_data = Data(x=torch.tensor(block_features, dtype=torch.float), edge_index=torch.tensor(block_edges, dtype=torch.long))
                acfg_list.append(one_acfg_data)
            
            item_function_names = item['function_names']
            item_function_edges = item['function_edges']
            
            local_function_name_list = item_function_names[:len(acfg_list)]
            assert len(acfg_list) == len(local_function_name_list), "The length of ACFG_List should be equal to the length of Local_Function_List"
            external_function_name_list = item_function_names[len(acfg_list):]
            
            external_function_index_list = [vocab[f_name] for f_name in external_function_name_list]
            index += 1
            torch.save(Data(hash=item_hash, local_acfgs=acfg_list, external_list=external_function_index_list, function_edges=item_function_edges, targets=label), "./{}.pt".format(index))
            print(index)


if __name__ == '__main__':
    json_path = "./sample.jsonl"
    train_vocab_file = "../ReservedDataCode/processed_dataset/train_external_function_name_vocab.jsonl"
    max_vocab_size = 10000
    vocabulary = Vocab(freq_file=train_vocab_file, max_vocab_size=max_vocab_size)
    parse_json_list_2_pyg_object(jsonl_file=json_path, label=1, vocab=vocabulary)