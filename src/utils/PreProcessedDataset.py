import os
import os.path as osp
from datetime import datetime

import torch
from torch_geometric.data import Dataset, DataLoader
from utils.RealBatch import create_real_batch_data  # noqa


class MalwareDetectionDataset(Dataset):
    def __init__(self, root, train_or_test, transform=None, pre_transform=None):
        super(MalwareDetectionDataset, self).__init__(None, transform, pre_transform)
        self.flag = train_or_test.lower()
        self.malware_root = os.path.join(root, "{}_malware".format(self.flag))
        self.benign_root = os.path.join(root, "{}_benign".format(self.flag))
        self.malware_files = os.listdir(self.malware_root)
        self.benign_files = os.listdir(self.benign_root)
    
    @staticmethod
    def _list_files_for_pt(the_path):
        files = []
        for name in os.listdir(the_path):
            if os.path.splitext(name)[-1] == '.pt':
                files.append(name)
        return files
    
    def __len__(self):
        # def len(self):
        # return 201
        return len(self.malware_files) + len(self.benign_files)
    
    def get(self, idx):
        split = len(self.malware_files)
        # split = 100
        if idx < split:
            idx_data = torch.load(osp.join(self.malware_root, 'malware_{}.pt'.format(idx)))
        else:
            over_fit_idx = idx - split
            idx_data = torch.load(osp.join(self.benign_root, "benign_{}.pt".format(over_fit_idx)))
        return idx_data


def _simulating(_dataset, _batch_size: int):
    print("\nBatch size = {}".format(_batch_size))
    time_start = datetime.now()
    print("start time: " + time_start.strftime("%Y-%m-%d@%H:%M:%S"))
    
    # https://github.com/pytorch/fairseq/issues/1560
    # https://github.com/pytorch/pytorch/issues/973#issuecomment-459398189
    # loaders_1 = DataLoader(dataset=benign_exe_dataset, batch_size=10, shuffle=True, num_workers=0)
    # increasing the shared memory: ulimit -SHn 51200
    loader = DataLoader(dataset=_dataset, batch_size=_batch_size, shuffle=True)  # default of prefetch_factor = 2 # num_workers=4
    
    for index, data in enumerate(loader):
        if index >= 3:
            break
        _real_batch, _position, _hash, _external_list, _function_edges, _true_classes = create_real_batch_data(one_batch=data)
        print(data)
        print("Hash: ", _hash)
        print("Position: ", _position)
        print("\n")
    
    time_end = datetime.now()
    print("end time: " + time_end.strftime("%Y-%m-%d@%H:%M:%S"))
    print("All time = {}\n\n".format(time_end - time_start))


if __name__ == '__main__':
    root_path: str = '/home/xiang/MalGraph/data/processed_dataset/DatasetJSON/'
    i_batch_size = 2
    
    train_dataset = MalwareDetectionDataset(root=root_path, train_or_test='train')
    print(train_dataset.malware_root, train_dataset.benign_root)
    print(len(train_dataset.malware_files), len(train_dataset.benign_files), len(train_dataset))
    _simulating(_dataset=train_dataset, _batch_size=i_batch_size)
    
    valid_dataset = MalwareDetectionDataset(root=root_path, train_or_test='valid')
    print(valid_dataset.malware_root, valid_dataset.benign_root)
    print(len(valid_dataset.malware_files), len(valid_dataset.benign_files), len(valid_dataset))
    _simulating(_dataset=valid_dataset, _batch_size=i_batch_size)
    
    test_dataset = MalwareDetectionDataset(root=root_path, train_or_test='test')
    print(test_dataset.malware_root, test_dataset.benign_root)
    print(len(test_dataset.malware_files), len(test_dataset.benign_files), len(test_dataset))
    _simulating(_dataset=test_dataset, _batch_size=i_batch_size)