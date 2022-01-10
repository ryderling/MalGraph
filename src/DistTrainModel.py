import logging
import math
import os
import random
from datetime import datetime

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as torch_mp
import torch.utils.data
import torch.utils.data.distributed
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from prefetch_generator import BackgroundGenerator
from sklearn.metrics import roc_auc_score, roc_curve
from torch import nn
from torch_geometric.data import DataLoader
from tqdm import tqdm

from models.HierarchicalGraphModel import HierarchicalGraphNeuralNetwork
from utils.FunctionHelpers import write_into, params_print_log, find_threshold_with_fixed_fpr
from utils.ParameterClasses import ModelParams, TrainParams, OptimizerParams, OneEpochResult
from utils.PreProcessedDataset import MalwareDetectionDataset
from utils.RealBatch import create_real_batch_data
from utils.Vocabulary import Vocab


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def reduce_sum(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)  # noqa
    return rt


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)  # noqa
    rt /= nprocs
    return rt


def all_gather_concat(tensor):
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


def train_one_epoch(local_rank, train_loader, valid_loader, model, criterion, optimizer, nprocs, idx_epoch, best_auc, best_model_file, original_valid_length, result_file):
    model.train()
    local_device = torch.device("cuda", local_rank)
    write_into(file_name_path=result_file, log_str="The local device = {} among {} nprocs in the {}-th epoch.".format(local_device, nprocs, idx_epoch))
    
    until_sum_reduced_loss = 0.0
    smooth_avg_reduced_loss_list = []
    
    for _idx_bt, _batch in enumerate(tqdm(train_loader, desc="reading _batch from local_rank={}".format(local_rank))):
        model.train()
        _real_batch, _position, _hash, _external_list, _function_edges, _true_classes = create_real_batch_data(one_batch=_batch)
        if _real_batch is None:
            write_into(result_file, "{}\n_real_batch is None in creating the real batch data of training ... ".format("*-" * 100))
            continue
        
        _real_batch = _real_batch.to(local_device)
        _position = torch.tensor(_position, dtype=torch.long).cuda(local_rank, non_blocking=True)
        _true_classes = _true_classes.float().cuda(local_rank, non_blocking=True)
        
        train_batch_pred = model(real_local_batch=_real_batch, real_bt_positions=_position, bt_external_names=_external_list, bt_all_function_edges=_function_edges, local_device=local_device)
        train_batch_pred = train_batch_pred.squeeze()
        
        loss = criterion(train_batch_pred, _true_classes)
        
        torch.distributed.barrier()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        reduced_loss = reduce_mean(loss, nprocs)
        until_sum_reduced_loss += reduced_loss.item()
        smooth_avg_reduced_loss_list.append(until_sum_reduced_loss / (_idx_bt + 1))
        
        if _idx_bt != 0 and (_idx_bt % math.ceil(len(train_loader) / 3) == 0 or _idx_bt == int(len(train_loader) - 1)):
            
            val_start_time = datetime.now()
            if local_rank == 0:
                write_into(result_file, "\nIn {}-th epoch, {}-th batch, we start to validate ... ".format(idx_epoch, _idx_bt))
            
            _eval_flag = "Valid_In_Train_Epoch_{}_Batch_{}".format(idx_epoch, _idx_bt)
            valid_result = validate(local_rank=local_rank, valid_loader=valid_loader, model=model, criterion=criterion, evaluate_flag=_eval_flag, distributed=True, nprocs=nprocs,
                                    original_valid_length=original_valid_length, result_file=result_file, details=False)
            
            if best_auc < valid_result.ROC_AUC_Score:
                _info = "[AUC Increased!] In evaluation of epoch-{} / batch-{}: AUC increased from {:.5f} < {:.5f}! Saving the model into {}".format(idx_epoch,
                                                                                                                                                     _idx_bt,
                                                                                                                                                     best_auc,
                                                                                                                                                     valid_result.ROC_AUC_Score,
                                                                                                                                                     best_model_file)
                best_auc = valid_result.ROC_AUC_Score
                torch.save(model.module.state_dict(), best_model_file)
            else:
                _info = "[AUC NOT Increased!] AUC decreased from {:.5f} to {:.5f}!".format(best_auc, valid_result.ROC_AUC_Score)
            
            if local_rank == 0:
                write_into(result_file, valid_result.__str__())
                write_into(result_file, _info)
                write_into(result_file, "[#One Validation Time#] Consume about {} time period for one validation.".format(datetime.now() - val_start_time))
    
    return smooth_avg_reduced_loss_list, best_auc


def validate(local_rank, valid_loader, model, criterion, evaluate_flag, distributed, nprocs, original_valid_length, result_file, details):
    model.eval()
    if distributed:
        local_device = torch.device("cuda", local_rank)
    else:
        local_device = torch.device("cuda")
    
    sum_loss = torch.tensor(0.0, dtype=torch.float, device=local_device)
    n_samples = torch.tensor(0, dtype=torch.int, device=local_device)
    
    all_true_classes = []
    all_positive_probs = []
    
    with torch.no_grad():
        for idx_batch, data in enumerate(tqdm(valid_loader)):
            _real_batch, _position, _hash, _external_list, _function_edges, _true_classes = create_real_batch_data(one_batch=data)
            if _real_batch is None:
                write_into(result_file, "{}\n_real_batch is None in creating the real batch data of validation ... ".format("*-" * 100))
                continue
            _real_batch = _real_batch.to(local_device)
            _position = torch.tensor(_position, dtype=torch.long).cuda(local_rank, non_blocking=True)
            _true_classes = _true_classes.float().cuda(local_rank, non_blocking=True)
            
            batch_pred = model(real_local_batch=_real_batch, real_bt_positions=_position, bt_external_names=_external_list, bt_all_function_edges=_function_edges, local_device=local_device)
            batch_pred = batch_pred.squeeze(-1)
            loss = criterion(batch_pred, _true_classes)
            sum_loss += loss.item()
            
            n_samples += len(batch_pred)
            
            all_true_classes.append(_true_classes)
            all_positive_probs.append(batch_pred)
    
    avg_loss = sum_loss / (idx_batch + 1)
    all_true_classes = torch.cat(all_true_classes, dim=0)
    all_positive_probs = torch.cat(all_positive_probs, dim=0)
    
    if distributed:
        torch.distributed.barrier()
        reduced_n_samples = reduce_sum(n_samples)
        reduced_avg_loss = reduce_mean(avg_loss, nprocs)
        gather_true_classes = all_gather_concat(all_true_classes).detach().cpu().numpy()
        gather_positive_prods = all_gather_concat(all_positive_probs).detach().cpu().numpy()
        
        gather_true_classes = gather_true_classes[:original_valid_length]
        gather_positive_prods = gather_positive_prods[:original_valid_length]
    
    else:
        reduced_n_samples = n_samples
        reduced_avg_loss = avg_loss
        gather_true_classes = all_true_classes.detach().cpu().numpy()
        gather_positive_prods = all_positive_probs.detach().cpu().numpy()
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
    _roc_auc_score = roc_auc_score(y_true=gather_true_classes, y_score=gather_positive_prods)
    _fpr, _tpr, _thresholds = roc_curve(y_true=gather_true_classes, y_score=gather_positive_prods)
    if details is True:
        _100_info = find_threshold_with_fixed_fpr(y_true=gather_true_classes, y_pred=gather_positive_prods, fpr_target=0.01)
        _1000_info = find_threshold_with_fixed_fpr(y_true=gather_true_classes, y_pred=gather_positive_prods, fpr_target=0.001)
    else:
        _100_info, _1000_info = "None", "None"
    
    _eval_result = OneEpochResult(Epoch_Flag=evaluate_flag, Number_Samples=reduced_n_samples, Avg_Loss=reduced_avg_loss, Info_100=_100_info, Info_1000=_1000_info,
                                  ROC_AUC_Score=_roc_auc_score, Thresholds=_thresholds, TPRs=_tpr, FPRs=_fpr)
    return _eval_result


def main_train_worker(local_rank: int, nprocs: int, train_params: TrainParams, model_params: ModelParams, optimizer_params: OptimizerParams, global_log: logging.Logger,
                      log_result_file: str):
    # dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:12345', world_size=nprocs, rank=local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=nprocs, rank=local_rank)
    torch.cuda.set_device(local_rank)
    
    # model configure
    vocab = Vocab(freq_file=train_params.external_func_vocab_file, max_vocab_size=train_params.max_vocab_size)
    
    if model_params.ablation_models.lower() == "full":
        model = HierarchicalGraphNeuralNetwork(model_params=model_params, external_vocab=vocab, global_log=global_log)
    else:
        raise NotImplementedError
    
    model.cuda(local_rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    
    if local_rank == 0:
        write_into(file_name_path=log_result_file, log_str=model.__str__())
    
    # loss function
    criterion = nn.BCELoss().cuda(local_rank)
    
    lr = optimizer_params.lr
    if optimizer_params.optimizer_name.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_params.optimizer_name.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=optimizer_params.weight_decay)
    else:
        raise NotImplementedError
    
    max_epochs = train_params.max_epochs
    
    dataset_root_path = train_params.processed_files_path
    train_batch_size = train_params.train_bs
    test_batch_size = train_params.test_bs
    
    # training dataset & dataloader
    train_dataset = MalwareDetectionDataset(root=dataset_root_path, train_or_test="train")
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoaderX(dataset=train_dataset, batch_size=train_batch_size, shuffle=False, num_workers=0, pin_memory=True, sampler=train_sampler)
    # validation dataset & dataloader
    valid_dataset = MalwareDetectionDataset(root=dataset_root_path, train_or_test="valid")
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
    valid_loader = DataLoaderX(dataset=valid_dataset, batch_size=test_batch_size, pin_memory=True, sampler=valid_sampler)
    
    if local_rank == 0:
        write_into(file_name_path=log_result_file, log_str="Training dataset={}, sampler={}, loader={}".format(len(train_dataset), len(train_sampler), len(train_loader)))
        write_into(file_name_path=log_result_file, log_str="Validation dataset={}, sampler={}, loader={}".format(len(valid_dataset), len(valid_sampler), len(valid_loader)))
    
    best_auc = 0.0
    ori_valid_length = len(valid_dataset)
    best_model_path = os.path.join(os.getcwd(), 'LocalRank_{}_best_model.pt'.format(local_rank))
    
    all_batch_avg_smooth_loss_list = []
    for epoch in range(max_epochs):
        train_sampler.set_epoch(epoch)
        valid_sampler.set_epoch(epoch)
        
        # train for one epoch
        time_start = datetime.now()
        if local_rank == 0:
            write_into(log_result_file, "\n{} start of {}-epoch, init best_auc={}, start time={} {}".format("-" * 50, epoch, best_auc, time_start.strftime("%Y-%m-%d@%H:%M:%S"), "-" * 50))
        
        smooth_avg_reduced_loss_list, best_auc = train_one_epoch(local_rank=local_rank, train_loader=train_loader, valid_loader=valid_loader, model=model, criterion=criterion,
                                                                 optimizer=optimizer, nprocs=nprocs, idx_epoch=epoch, best_auc=best_auc, best_model_file=best_model_path,
                                                                 original_valid_length=ori_valid_length, result_file=log_result_file)
        all_batch_avg_smooth_loss_list.extend(smooth_avg_reduced_loss_list)
        
        # adjust learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / optimizer_params.learning_anneal
        
        time_end = datetime.now()
        if local_rank == 0:
            write_into(log_result_file, "\n{} end of {}-epoch, with best_auc={}, len of loss={}, end time={}, and time period={} {}".format("*" * 50, epoch, best_auc,
                                                                                                                                            len(smooth_avg_reduced_loss_list),
                                                                                                                                            time_end.strftime("%Y-%m-%d@%H:%M:%S"),
                                                                                                                                            time_end - time_start, "*" * 50))


# https://hydra.cc/docs/tutorials/basic/your_first_app/defaults#overriding-a-config-group-default
@hydra.main(config_path="../configs/", config_name="default.yaml")
def main_app(config: DictConfig):
    # set seed for determinism for reproduction
    random.seed(config.Training.seed)
    np.random.seed(config.Training.seed)
    torch.manual_seed(config.Training.seed)
    torch.cuda.manual_seed(config.Training.seed)
    torch.cuda.manual_seed_all(config.Training.seed)
    
    # setting hyper-parameter for Training / Model / Optimizer
    _train_params = TrainParams(processed_files_path=to_absolute_path(config.Data.preprocess_root), max_epochs=config.Training.max_epoches, train_bs=config.Training.train_batch_size, test_bs=config.Training.test_batch_size, external_func_vocab_file=to_absolute_path(config.Data.train_vocab_file), max_vocab_size=config.Data.max_vocab_size)
    _model_params = ModelParams(gnn_type=config.Model.gnn_type, pool_type=config.Model.pool_type, acfg_init_dims=config.Model.acfg_node_init_dims, cfg_filters=config.Model.cfg_filters, fcg_filters=config.Model.fcg_filters, number_classes=config.Model.number_classes, dropout_rate=config.Model.drapout_rate, ablation_models=config.Model.ablation_models)
    _optim_params = OptimizerParams(optimizer_name=config.Optimizer.name, lr=config.Optimizer.learning_rate, weight_decay=config.Optimizer.weight_decay, learning_anneal=config.Optimizer.learning_anneal)
    
    # logging
    log = logging.getLogger("DistTrainModel.py")
    log.setLevel("DEBUG")
    log.warning("Hydra's Current Working Directory: {}".format(os.getcwd()))
    
    # setting for the log directory
    result_file = '{}_{}_{}_ACFG_{}_FCG_{}_Epoch_{}_TrainBS_{}_LR_{}_Time_{}.txt'.format(_model_params.ablation_models, _model_params.gnn_type, _model_params.pool_type,
                                                                                         _model_params.cfg_filters, _model_params.fcg_filters, _train_params.max_epochs,
                                                                                         _train_params.train_bs, _optim_params.lr, datetime.now().strftime("%Y%m%d_%H%M%S"))
    log_result_file = os.path.join(os.getcwd(), result_file)
    
    _other_params = {"Hydra's Current Working Directory": os.getcwd(), "seed": config.Training.seed, "log result file": log_result_file, "only_test_path": config.Training.only_test_path}
    
    params_print_log(_train_params.__dict__, log_result_file)
    params_print_log(_model_params.__dict__, log_result_file)
    params_print_log(_optim_params.__dict__, log_result_file)
    params_print_log(_other_params, log_result_file)
    
    if config.Training.only_test_path == 'None':
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(config.Training.dist_port)
        # num_gpus = 1
        num_gpus = torch.cuda.device_count()
        log.info("Total number of GPUs = {}".format(num_gpus))
        torch_mp.spawn(main_train_worker, nprocs=num_gpus, args=(num_gpus, _train_params, _model_params, _optim_params, log, log_result_file,))
        
        best_model_file = os.path.join(os.getcwd(), 'LocalRank_{}_best_model.pt'.format(0))
    
    else:
        best_model_file = config.Training.only_test_path
    
    # model re-init and loading
    log.info("\n\nstarting to load the model & re-validation & testing from the file of \"{}\" \n".format(best_model_file))
    device = torch.device('cuda')
    train_vocab_path = _train_params.external_func_vocab_file
    vocab = Vocab(freq_file=train_vocab_path, max_vocab_size=_train_params.max_vocab_size)
    
    if _model_params.ablation_models.lower() == "full":
        model = HierarchicalGraphNeuralNetwork(model_params=_model_params, external_vocab=vocab, global_log=log)
    else:
        raise NotImplementedError
    model.to(device)
    model.load_state_dict(torch.load(best_model_file, map_location=device))
    criterion = nn.BCELoss().cuda()
    
    test_batch_size = config.Training.test_batch_size
    dataset_root_path = _train_params.processed_files_path
    # validation dataset & dataloader
    valid_dataset = MalwareDetectionDataset(root=dataset_root_path, train_or_test="valid")
    valid_dataloader = DataLoaderX(dataset=valid_dataset, batch_size=test_batch_size, shuffle=False)
    log.info("Total number of all validation samples = {} ".format(len(valid_dataset)))
    
    # testing dataset & dataloader
    test_dataset = MalwareDetectionDataset(root=dataset_root_path, train_or_test="test")
    test_dataloader = DataLoaderX(dataset=test_dataset, batch_size=test_batch_size, shuffle=False)
    log.info("Total number of all testing  samples = {} ".format(len(test_dataset)))
    
    _valid_result = validate(valid_loader=valid_dataloader, model=model, criterion=criterion, evaluate_flag="DoubleCheckValidation", distributed=False, local_rank=None, nprocs=None, original_valid_length=len(valid_dataset), result_file=log_result_file, details=True)
    log.warning("\n\n" + _valid_result.__str__())
    _test_result = validate(valid_loader=test_dataloader, model=model, criterion=criterion, evaluate_flag="FinalTestingResult", distributed=False, local_rank=None, nprocs=None, original_valid_length=len(test_dataset), result_file=log_result_file, details=True)
    log.warning("\n\n" + _test_result.__str__())


if __name__ == '__main__':
    main_app()