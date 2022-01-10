from dataclasses import dataclass


@dataclass
class TrainParams:
    processed_files_path: str
    # train_test_split_file: str
    max_epochs: int
    train_bs: int
    test_bs: int
    external_func_vocab_file: str
    max_vocab_size: int


@dataclass
class OptimizerParams:
    optimizer_name: str
    lr: float
    weight_decay: float
    learning_anneal: float


@dataclass
class ModelParams:
    gnn_type: str
    pool_type: str
    acfg_init_dims: int
    cfg_filters: str
    fcg_filters: str
    number_classes: int
    dropout_rate: float
    ablation_models: str


@dataclass
class OneEpochResult:
    Epoch_Flag: str
    Number_Samples: int
    Avg_Loss: float
    Info_100: str
    Info_1000: str
    ROC_AUC_Score: float
    Thresholds: list
    TPRs: list
    FPRs: list
    
    def __str__(self):
        s = "\nResult of \"{}\":\n=Epoch_Flag = {}\n=>Number of samples = {}\n=>Avg_Loss = {}\n=>Info_100 = {}\n=>Info_1000 = {}\n=>ROC_AUC_score = {}\n".format(
            self.Epoch_Flag,
            self.Epoch_Flag,
            self.Number_Samples,
            self.Avg_Loss,
            self.Info_100,
            self.Info_1000,
            self.ROC_AUC_Score)
        return s


if __name__ == '__main__':
    pass