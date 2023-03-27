import torch
import json
import numpy as np
from skorch.regressor import NeuralNetRegressor
from skorch.classifier import NeuralNetClassifier
from skorch.dataset import Dataset as SkDataset
from skorch.callbacks import EarlyStopping, EpochScoring
from skorch.helper import predefined_split
from skorch.utils import multi_indexing
from torch.optim import AdamW
from torch.nn import MSELoss, BCEWithLogitsLoss, CrossEntropyLoss
from sklearn.metrics import r2_score, f1_score
from auto_search.rtdl_modules import MLP
from auto_search.utils import calculate_metrics, MetricsReport, TaskType, PredictionType
from entityencoder_utils.general_utils import tensor2ndarray


class CustomizedSkDatasetForAE(SkDataset):
    def __init__(self, X, y):
        super(CustomizedSkDatasetForAE, self).__init__(X=X, y=y)


def rtdl_evaluation(raw_data: dict, d_in: int, d_out: int, ds_info: dict, encoder_wrapper):

    device = encoder_wrapper.device

    with open('../auto_search/configs/adult_cv.json') as f:
        config = json.load(f)
    assert config is not None

    mlp_params = {}
    mlp_params["d_layers"] = config['d_layers']
    mlp_params["dropout"] = config['dropout']
    mlp_params["d_in"] = d_in
    mlp_params["d_out"] = d_out

    mlp = MLP.make_baseline(**mlp_params)

    if ds_info['n_classes'] > 2:
        y_train = raw_data['y_train'].numpy().reshape(-1).astype(np.int64)
        y_val = raw_data['y_val'].numpy().reshape(-1).astype(np.int64)
        y_test = raw_data['y_test'].reshape(-1).astype(np.int64)
    else:
        y_train = raw_data['y_train'].numpy().reshape(-1, 1).astype(np.float32)
        y_val = raw_data['y_val'].numpy().reshape(-1, 1).astype(np.float32)
        y_test = raw_data['y_test'].reshape(-1, 1).astype(np.float32)

    X_train  = raw_data['X_train']
    X_val = raw_data['X_val']
    X_test = raw_data['X_test_tensor']
    X_train = tensor2ndarray(encoder_wrapper.encode(X_train))
    X_val = tensor2ndarray(encoder_wrapper.encode(X_val))
    X_test = tensor2ndarray(encoder_wrapper.encode(X_test))

    train_ds = CustomizedSkDatasetForAE(X=X_train, y=y_train)
    val_ds = CustomizedSkDatasetForAE(X=X_val, y=y_val)
    es = EarlyStopping(monitor='valid_loss', patience=16)

    def f1(net, X, y):
        y_pred = net.predict(X)
        return f1_score(y, y_pred, average='macro')

    def r2(net, X, y):
        y_pred = net.predict(X)
        return r2_score(y, y_pred)

    if ds_info['is_regression']:
        net = NeuralNetRegressor(
            mlp,
            criterion=MSELoss,
            optimizer=AdamW,
            lr=config['lr'],
            optimizer__weight_decay=config['weight_decay'],
            batch_size=128 if len(raw_data['y_train']) < 10_000 else 256,
            max_epochs=1000,
            train_split=predefined_split(val_ds),
            iterator_train__shuffle=True,
            device=device,
            callbacks=[es, EpochScoring(r2, lower_is_better=False)],
        )

    else:
        net = NeuralNetClassifier(
            mlp,
            criterion=BCEWithLogitsLoss if ds_info['n_classes'] == 2 else CrossEntropyLoss,
            optimizer=AdamW,
            lr=config['lr'],
            optimizer__weight_decay=config['weight_decay'],
            batch_size=128 if len(raw_data['y_train']) < 10_000 else 256,
            max_epochs=1000,
            train_split=predefined_split(val_ds),
            iterator_train__shuffle=True,
            device=device,
            callbacks=[es, EpochScoring(f1, lower_is_better=False)],
        )

    net.fit(
        X=train_ds.X,
        y=train_ds.y
    )

    print('LAST:', len(net.history))

    X_splits = {}
    X_splits["train"] = X_train
    X_splits["val"] = X_val
    X_splits["test"] = X_test

    predictions = {
        k: net.predict_proba(v)[:, 1] if ds_info['n_classes'] == 2 else
        net.predict_proba(v) if ds_info['n_classes'] > 2 else
        net.predict(v)
        for k, v in X_splits.items()
    }

    if ds_info['is_regression']:
        task_type = TaskType('regression')
        prediction_type = None
    elif ds_info['n_classes'] == 2:
        task_type = TaskType('binclass')
        prediction_type = PredictionType('probs')
    else:
        task_type = TaskType('multiclass')
        prediction_type = PredictionType('probs')

    report = {}
    report['metrics'] = {'test': calculate_metrics(y_true=y_test,
                                                   y_pred=predictions['test'],
                                                   task_type=task_type,
                                                   prediction_type=prediction_type,
                                                   y_info={'std': 1.0} if ds_info['is_regression'] else {}),

                         'val': calculate_metrics(y_true=y_val,
                                                  y_pred=predictions['val'],
                                                  task_type=task_type,
                                                  prediction_type=prediction_type,
                                                  y_info={'std': 1.0} if ds_info['is_regression'] else {}),

                         'train': calculate_metrics(y_true=y_train,
                                                    y_pred=predictions['train'],
                                                    task_type=task_type,
                                                    prediction_type=prediction_type,
                                                    y_info={'std': 1.0} if ds_info['is_regression'] else {})}

    metrics_report = MetricsReport(report['metrics'], task_type)
    metrics_report.print_metrics()

    if ds_info['is_regression']:
        return metrics_report._res['test']['r2']

    return metrics_report._res['test']['f1']
