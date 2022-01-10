import logging
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict

from sklearn.metrics import auc, confusion_matrix, balanced_accuracy_score
from texttable import Texttable
from datetime import datetime


def only_get_fpr(y_true, y_pred):
    n_benign = (y_true == 0).sum()
    n_false = (y_pred[y_true == 0] == 1).sum()
    return float(n_false) / float(n_benign)


def get_fpr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()
    return float(fp) / float(fp + tn)


def find_threshold_with_fixed_fpr(y_true, y_pred, fpr_target):
    start_time = datetime.now()
    
    threshold = 0.0
    fpr = only_get_fpr(y_true, y_pred > threshold)
    while fpr > fpr_target and threshold <= 1.0:
        threshold += 0.0001
        fpr = only_get_fpr(y_true, y_pred > threshold)
    
    tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred > threshold).ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    acc = (tp + tn) / (tn + fp + fn + tp)  # equal to accuracy_score(y_true=y_true, y_pred=y_pred > threshold)
    balanced_acc = balanced_accuracy_score(y_true=y_true, y_pred=y_pred > threshold)
    
    _info = "Threshold: {:.6f}, TN: {}, FP: {}, FN: {}, TP: {}, TPR: {:.6f}, FPR: {:.6f}, ACC: {:.6f}, Balanced_ACC: {:.6f}. consume about {} time in find threshold".format(
        threshold, tn, fp, fn, tp, tpr, fpr, acc, balanced_acc, datetime.now() - start_time)
    return _info


def alphabet_lower_strip(str1):
    return re.sub("[^A-Za-z]", "", str1).lower()


def filter_counter_with_threshold(counter: Counter, min_threshold):
    return {x: counter[x] for x in counter if counter[x] >= min_threshold}


def create_dir_if_not_exists(new_dir: str, log: logging.Logger):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
        log.info('We are creating the dir of \"{}\" '.format(new_dir))
    else:
        log.info('We CANNOT creat the dir of \"{}\" as it is already exists.'.format(new_dir))


def get_jsonl_files_from_path(file_path: str, log: logging.Logger):
    file_list = []
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if file.endswith(".jsonl"):
                file_list.append(os.path.join(root, file))
    file_list.sort()
    log.info("{}\nFrom the path of {}, we obtain a list of {} files as follows.".format("-" * 50, file_path, len(file_list)))
    log.info("\n" + '\n'.join(str(f) for f in file_list))
    return file_list


def write_into(file_name_path: str, log_str: str, print_flag=True):
    if print_flag:
        print(log_str)
    if log_str is None:
        log_str = 'None'
    if os.path.isfile(file_name_path):
        with open(file_name_path, 'a+') as log_file:
            log_file.write(log_str + '\n')
    else:
        with open(file_name_path, 'w+') as log_file:
            log_file.write(log_str + '\n')


def params_print_log(param_dict: Dict, log_path: str):
    keys = sorted(param_dict.keys())
    table = Texttable()
    table.set_precision(6)
    table.set_cols_align(["l", "l", "c"])
    table.add_row(["Index", "Parameters", "Values"])
    for index, k in enumerate(keys):
        table.add_row([index, k, str(param_dict[k])])
    
    # print(table.draw())
    write_into(file_name_path=log_path, log_str=table.draw())


def dataclasses_to_string(ins: dataclass):
    name = type(ins).__name__
    
    var_list = [f"{key} = {value}" for key, value in vars(ins).items()]
    var_str = '\n=>'.join(var_list)
    
    return f"{name}:\n=>{var_str}\n"


if __name__ == '__main__':
    pass