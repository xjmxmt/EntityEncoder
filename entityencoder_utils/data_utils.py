import os
import pickle

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from collections import Counter
import torch
import time

from auto_search.tune_ae_cat import get_best_hyperparams, objective
from entityencoder_utils.general_utils import hardcoded_log
from entity_encoders.encoder import Encoder


def preprocess(dataset, save=False) -> None:
    if dataset == 'buddy':
        path = "../data/buddy/train.csv"

        # get_dataset(dataset, path=path)

        # some preprocessing steps
        df = pd.read_csv(path)
        k = ['issue_date', 'listing_date']
        for i in k:
            df[i] = df[i].apply(lambda x: time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S')))
        df['condition'] = df['condition'].fillna(3.0)

        if save:
            df.to_csv("../data/buddy/processed_train.csv", index=False)
    elif dataset == 'king':
        path = "../data/king/kc_house_data.csv"
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date']).apply(lambda x: time.mktime(time.strptime(str(x), '%Y-%m-%d %H:%M:%S')))

        if save:
            df.to_csv("../data/king/processed_train.csv", index=False)
    elif dataset == 'cover':
        path = "../data/cover/cover.csv"
        df = pd.read_csv(path)
        df['Soil'] = (df.iloc[:, 14:54] == 1).idxmax(1)
        df['Wilderness'] = (df.iloc[:, 10:14] == 1).idxmax(1)
        df['Cover_Type'] -= 1
        cols = df.columns[:10].tolist()
        cols += ['Wilderness', 'Soil', 'Cover_Type']
        df = df[cols]

        if save:
            df.to_csv("../data/cover/processed_train.csv", index=False)
    elif dataset == 'loan':
        path = "../data/loan/loan.csv"
        df = pd.read_csv(path)

        # fill na in categorical columns
        df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
        df['Married'].fillna(df['Married'].mode()[0], inplace=True)
        df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
        df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
        df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
        df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)

        # fill na in numerical columns
        df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)

        if save:
            df.to_csv("../data/loan/processed_train.csv", index=False)


def get_dataset(
        path,
        dataset_info,
        parent_dir,
        encoding='ae_cat',
        decoding='mlp',
        is_y_cond=True,
        epochs=2000,
        batch_size=4096,
        lr=1e-3,
        auto_search=True,
        best_params=None,
        return_encoder=True,
        seed=0,
        device=torch.device('cuda')
):
    """
    Apply data preprocessing to raw .csv datasets

    :param ds_name: str
    :param path: str
    :param encoding: str, describe the way to transfer (discrete) features into a continuous space
        'pca-pc-cat':       PCA,            per column,     categorical features only,                (one-hot encoder) TODO
        'pca-whole-cat':    PCA,            all columns,    categorical features only,                (ordinal encoder) TODO
        'pca-whole-both':   PCA,            all columns,    both categorical and numerical features,  (ordinal encoder) TODO
        'ae-whole-both':    AE (CE+MSE),    all columns,    both categorical and numerical features,  (ordinal encoder)
        'ae_cat':     AE (CE),        all columns,    categorical features only,                (ordinal encoder)
        'nn-whole-cat':     Supervised,     all column,     categorical features only,                (ordinal encoder)
    :param is_y_cond: boolean, decide whether to include label y
    :param seed: boolean, seed for data shuffle & QuantileTransformer
    :return: dataset_info: dict
    """

    assert path.endswith('csv')
    assert encoding in ['pca_cat', 'ae_both', 'ae_cat', 'nn_cat']
    print(f'log: start data preprocessing on file: {path}')
    print(f'log: start data preprocessing with method: {encoding}')

    including_y = False
    if not dataset_info['is_regression'] and not is_y_cond:
        including_y = True

    df = pd.read_csv(path)
    y_col = dataset_info['y_col']
    drop_cols = dataset_info['drop_cols']

    X = df.drop(labels=drop_cols, axis=1)
    Y = df[y_col]

    y_label_encoder = None
    if Y.dtype == object:
        y_label_encoder = sklearn.preprocessing.LabelEncoder()
        Y = pd.DataFrame(y_label_encoder.fit_transform(Y.values))

    categorical_cols = dataset_info['categorical_cols']
    numerical_cols = [i for i in X.columns if i not in categorical_cols]
    n_cat = len(categorical_cols)
    n_num = len(numerical_cols)
    print(f'log: num of categorical features: {n_cat}, num of numerical features: {n_num}')

    # Note: all dataset should have the order that categorical features comes first
    order = categorical_cols + numerical_cols
    X = X[order]

    X, X_test, Y, y_test = train_test_split(X, Y, test_size=0.2, random_state=seed, shuffle=True)

    embedded_cols = [len(Counter(X[col])) for col in categorical_cols]
    if including_y:
        embedded_cols.append(dataset_info['n_classes'])

    if encoding == 'nn_cat' or encoding == 'ae_both' or encoding == 'ae_cat':
        embedding_sizes = [(n_categories, hardcoded_log(n_categories)) for n_categories in embedded_cols]
    else:
        embedding_sizes = []

    print(f'log: embedding_sizes: {embedding_sizes}')

    cat_encoder = None
    if n_cat != 0:
        if encoding == 'pca_cat':
            raise "Not Implemented."
        else:
            cat_encoder = sklearn.preprocessing.OrdinalEncoder()
            X_cat = cat_encoder.fit_transform(X.loc[:, categorical_cols].values)

            if including_y:
                X_cat = np.hstack([X_cat, Y.values.reshape(-1, 1)])

        print(f'log: categorical encoder: {cat_encoder}, categorical columns shape: {X_cat.shape}')

    normalizer = None
    if n_num != 0:
        X_num = X.loc[:, numerical_cols].values
        if dataset_info['is_regression'] and not is_y_cond:
            X_num = np.hstack([X_num, Y.values.reshape(-1, 1)])

        # Note: refer to TabDDPM
        normalizer = sklearn.preprocessing.QuantileTransformer(
            output_distribution="normal",
            n_quantiles=max(min(X.shape[0] // 30, 1000), 10),
            subsample=1e9,
            random_state=seed,
        )
        X_num = normalizer.fit_transform(X_num)

    if n_cat != 0 and n_num != 0:
        X = pd.DataFrame(data=np.hstack([X_cat, X_num]))
    elif n_cat != 0:
        X = pd.DataFrame(data=X_cat)
    else:
        X = pd.DataFrame(data=X_num)

    if n_cat != 0:
        if encoding == 'pca_cat':
            raise "Unexpected encoding type."
            # n_cat_emb = embedding_sizes[-1][-1]
        else:
            n_cat_emb = sum([i for n, i in embedding_sizes])
    else:
        n_cat_emb = 0
    print(f'log: dim of concatenated cat embeddings: {n_cat_emb}')

    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=seed, shuffle=True)

    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.int64)
    print("log: shape of X_train after preprocessing: ", X_train.shape, "shape of y_train: ", y_train.shape)
    X_val = torch.tensor(X_val.values, dtype=torch.float32)
    y_val = torch.tensor(y_val.values, dtype=torch.int64)
    print("log: shape of X_val after preprocessing: ", X_val.shape, "shape of y_val: ", y_val.shape)

    if n_cat != 0:
        X_test_cat = X_test.loc[:, categorical_cols].values
        # add category appears in test set
        n_col = X_test_cat.shape[1]
        add_category_func = np.frompyfunc(lambda s, idx: cat_encoder.categories_[idx][0] if s not in cat_encoder.categories_[idx] else s, nin=2, nout=1)
        for i in range(n_col):
            X_test_cat[:, i] = add_category_func(X_test_cat[:, i], i)
            # Note: using -1 to represent unseen label
            # if -1 in X_test_cat[:, i]:
            #     cat_encoder.categories_[i] = np.append(cat_encoder.categories_[i], -1)
        X_test_cat = cat_encoder.transform(X_test_cat)

        if including_y:
            X_test_cat = np.hstack([X_test_cat, y_test.values.reshape(-1, 1)])

    if n_num != 0:
        X_test_num = X_test.loc[:, numerical_cols].values
        if dataset_info['is_regression'] and not is_y_cond:
            X_test_num = np.hstack([X_test_num, y_test.values.reshape(-1, 1)])
        X_test_num = normalizer.transform(X_test_num)

    if n_cat != 0 and n_num != 0:
        processed_X_test = np.hstack([X_test_cat, X_test_num])
    elif n_cat != 0:
        processed_X_test = X_test_cat
    else:
        processed_X_test = X_test_num
    processed_X_test = torch.tensor(processed_X_test, dtype=torch.float32)
    processed_y_test = torch.tensor(y_test.values, dtype=torch.int64)
    print("log: shape of X_test after preprocessing: ", processed_X_test.shape, "shape of y_val: ", processed_y_test.shape)

    data = {
        "X_train": X_train,  # torch.Tensor
        "y_train": y_train,  # torch.Tensor
        "X_val": X_val,  # torch.Tensor
        "y_val": y_val,  # torch.Tensor
        "X_test": X_test.values,  # numpy.ndarray
        "y_test": y_test.values,  # numpy.ndarray
        "X_test_tensor": processed_X_test,  # torch.Tensor
        "y_test_tensor": processed_y_test  # torch.Tensor
    }

    preprocessors = {
        "cat_encoder": cat_encoder,
        "y_label_encoder": y_label_encoder,
        "normalizer": normalizer
    }

    processed_dataset_info = {
        "embedding_sizes": embedding_sizes,
        "n_cat_emb": n_cat_emb,
        "n_cat": n_cat,
        "n_num": n_num,
        "is_regression": dataset_info['is_regression'],
        "is_y_cond": is_y_cond,
        "including_y": including_y,
        "n_classes": dataset_info['n_classes']
    }

    if return_encoder and auto_search:
        best_trail = get_best_hyperparams(objective, raw_data=data, ds_info=processed_dataset_info, parent_dir=parent_dir, metrics='distance')
        print(best_trail) # e.g. params={'lr': 0.0002305083086224178, 'epochs': 500, 'latent_dim': 'sqrt', 'n_layers': 3, 'using_noise': False, 'emb_activation': 'relu', 'cat_ratio': 0.5714285714285714}
        best_params = best_trail.params

        # best_params = {'lr': 0.00038533230648127927, 'epochs': 500, 'latent_dim': None, 'n_layers': 5, 'using_noise': True, 'emb_activation': None, 'cat_ratio': 0.5714285714285714}
        model_params = {}
        if 'latent_dim' in best_params:
            model_params['latent_dim'] = best_params['latent_dim']
        if 'n_layers' in best_params:
            model_params['n_layers'] = best_params['n_layers']
        if 'using_noise' in best_params:
            model_params['using_noise'] = best_params['using_noise']
        if 'emb_activation' in best_params:
            model_params['emb_activation'] = best_params['emb_activation']
        if 'cat_ratio' in best_params:
            model_params['cat_ratio'] = best_params['cat_ratio']
        if 'epochs' in best_params:
            epochs = best_params['epochs']
        if 'lr' in best_params:
            lr = best_params['lr']

        entity_encoder = Encoder(parent_dir, processed_dataset_info,
                                 encoding_type=encoding, decoding_type=decoding, encoding_y=including_y,  params=model_params,
                                 seed=seed, device=device)
        os.makedirs(parent_dir, exist_ok=True)
        entity_encoder.fit(X_train, y_train, parent_dir, epochs=epochs, batch_size=batch_size, lr=lr)

        print('best params: ', best_params)

        return data, preprocessors, processed_dataset_info, entity_encoder

    elif return_encoder:

        if best_params is not None:
            model_params = {}
            if 'latent_dim' in best_params:
                model_params['latent_dim'] = best_params['latent_dim']
            if 'n_layers' in best_params:
                model_params['n_layers'] = best_params['n_layers']
            if 'using_noise' in best_params:
                model_params['using_noise'] = best_params['using_noise']
            if 'emb_activation' in best_params:
                model_params['emb_activation'] = best_params['emb_activation']
            if 'cat_ratio' in best_params:
                model_params['cat_ratio'] = best_params['cat_ratio']
            if 'epochs' in best_params:
                epochs = best_params['epochs']
            if 'lr' in best_params:
                lr = best_params['lr']

            print('log: encoder params: ', model_params)

            entity_encoder = Encoder(parent_dir, processed_dataset_info,
                                     encoding_type=encoding, decoding_type=decoding, encoding_y=including_y,
                                     params=model_params, seed=seed, device=device)

        else:
            entity_encoder = Encoder(parent_dir, processed_dataset_info,
                                     encoding_type=encoding, decoding_type=decoding, encoding_y=including_y,
                                     seed=seed, device=device)

        if os.path.exists(os.path.join(parent_dir, "ae_weights.pt")):
            entity_encoder.wrapper.model.load_state_dict(
                torch.load(os.path.join(parent_dir, f"ae_weights.pt"), map_location=device))

            if encoding == 'ae_cat' and decoding == '1nn':
                with open(os.path.join(parent_dir, '1nn_weights.pickle'), 'rb') as f:
                    nn_classifiers = pickle.load(f)
                entity_encoder.wrapper.nn_classifiers = nn_classifiers

            print(f'log: Autoencoder weights loaded')
        else:
            os.makedirs(parent_dir, exist_ok=True)
            entity_encoder.fit(X_train, y_train, parent_dir, epochs=epochs, batch_size=batch_size, lr=lr)

        return data, preprocessors, processed_dataset_info, entity_encoder

    else:

        return data, preprocessors, processed_dataset_info, None


datasets_dict = {
    # https://www.kaggle.com/competitions/churn-modelling
    "churn": {"drop_labels": ["RowNumber", "CustomerId", "Surname", "Exited"],
              "categorical_labels": ["Geography", "Gender", "HasCrCard", "IsActiveMember"],
              "y_label": "Exited",
              "is_regression": False,
              "n_classes": 2},

    # https://archive.ics.uci.edu/ml/datasets/adult
    "adult": {"drop_labels": ["y"],
              "categorical_labels": ["6", "7", "8", "9", "10", "11", "12", "13"],
              "y_label": "y",
              "is_regression": False,
              "n_classes": 2},

    # https://www.hackerearth.com/en-us/challenges/competitive/hackerearth-machine-learning-challenge-pet-adoption/
    "buddy": {"drop_labels": ["pet_id", "breed_category"],
              "categorical_labels": ["condition", "color_type", "X1", "X2", "pet_category"],
              "continuous_labels": ["issue_date", "listing_date", "length(m)", "height(cm)"],
              "y_label": "breed_category",
              "is_regression": False,
              "n_classes": 3},

    # https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
    "king": {"drop_labels": ["id", "price"],
             "categorical_labels": ["view", "condition", "waterfront"],
             "y_label": "price",
             "is_regression": True,
             "n_classes": 1},

    # length:  39974
    "cover": {"drop_labels": ["Cover_Type"],
              "categorical_labels": ["Wilderness", "Soil"],
              "y_label": "Cover_Type",
              "is_regression": False,
              "n_classes": 7},

    # length:  40389
    "intrusion": {"drop_labels": ["class"],
                  "categorical_labels": ["protocol_type", "service", "flag", "land", "wrong_fragment",  "urgent", "hot",
                                         "num_failed_logins", "logged_in", "num_compromised", "root_shell",
                                         "su_attempted", "num_root", "num_file_creations", "num_shells",
                                         "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login"],
                  "y_label": "class",
                  "is_regression": False,
                  "n_classes": 23},

    # https://www.kaggle.com/datasets/burak3ergun/loan-data-set?resource=download
    # length: 614
    "loan": {"drop_labels": ["Loan_ID", "Loan_Status"],
              "categorical_labels": [ "Gender", "Married", "Dependents", "Education", "Self_Employed", "Credit_History",
                                      "Loan_Amount_Term", "Property_Area"],
              "y_label": "Loan_Status",
              "is_regression": False,
              "n_classes": 2},
}


if __name__ == "__main__":

    adult_info = {
        "drop_cols": ["y"],
        "categorical_cols": ["6", "7", "8", "9", "10", "11", "12", "13"],
        "y_col": "y",
        "is_regression": False,
        "n_classes": 2
    }

    intrusion_info = {
        "drop_cols": ["class"],
        "categorical_cols": ["protocol_type", "service", "flag", "land", "wrong_fragment", "urgent", "hot",
                               "num_failed_logins", "logged_in", "num_compromised", "root_shell",
                               "su_attempted", "num_root", "num_file_creations", "num_shells",
                               "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login"],
        "y_col": "class",
        "is_regression": False,
        "n_classes": 23}

    loan_info = {
        "drop_cols": ["Loan_ID", "Loan_Status"],
        "categorical_cols": ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Credit_History",
                             "Loan_Amount_Term", "Property_Area"],
        "y_col": "Loan_Status",
        "is_regression": False,
        "n_classes": 2
    }

    # test_path = "../test_cases/data/intrusion/processed_train.csv"
    # parent_dir = "../test_cases/exp/intrusion/99"

    test_path = "../test_cases/data/loan/processed_train.csv"
    parent_dir = "../test_cases/exp/loan/99"
    tuple = get_dataset(test_path, loan_info, parent_dir, epochs=100, encoding='ae_cat', decoding='mlp', is_y_cond=True)

    # # preprocess(dataset, save=True)
    #
    # dataset_info = get_dataset(dataset, path=path, encoding='ae-whole-cat')
    # x_train = dataset_info['X_train']
    # print(len(x_train))

    # path = "../data/cover/cover.csv"
    # dataset = 'cover'

    # df = pd.read_csv(path)
    # categorical = [col for col, dtype in df.convert_dtypes().dtypes.items() if str(dtype) in ["string", "Int64"]]
    #
    # print(categorical)
    # for col in categorical:
    #     print(col, len(Counter(df[col])), Counter(df[col]))
    #
    # print(df.info())

    # df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
    # df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    # df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
    # df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
    # df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
    # df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
    #
    # print(categorical)
    # for col in categorical:
    #     print(col, len(Counter(df[col])), Counter(df[col]))
    #
    # print(df.info())

    # print(df.isna().sum())
    # print(df.iloc[0, :])
    # print(df.info())
    # print('length: ', len(df))
    # cols = df.columns
    # selected_col = []
    # for col in cols:
    #     n = len(Counter(df[col]))
    #     print(col, n, df[col].dtype)
        # if n < 1000 and ((df[col].dtype == np.int64) or (df[col].dtype == object)):
        #     selected_col.append(col)
    # print(selected_col)
    # print(len(cols))

    # print(len(Counter(df['class'])), Counter(df['class']))

    # df.to_csv("../data/intrusion/processed_trained.csv", index=False)

    # dataset_info = get_dataset(dataset, path, encoding='ae-whole-cat')
    # print(Counter(dataset_info['y_train'].numpy()))
    # print(Counter(dataset_info['y_test']))

    # # Saving datasets as npy files
    # df = pd.read_csv(path)
    # label_info = datasets_dict[dataset]
    # categorical_labels = label_info['categorical_labels']
    # y_label = label_info['y_label']
    # drop_labels = label_info['drop_labels']
    #
    # X = df.drop(labels=drop_labels, axis=1)
    # Y = df[y_label]
    # label_encoder = sklearn.preprocessing.LabelEncoder()
    # Y = label_encoder.fit_transform(Y)
    #
    # n_cat = len(categorical_labels)
    # continuous_labels = [i for i in X.columns if i not in categorical_labels]
    # n_cont = len(continuous_labels)
    # print("n_cat, n_cont: ", n_cat, n_cont)
    #
    # print(categorical_labels)
    # for col in categorical_labels:
    #     print(col, len(Counter(df[col])), Counter(df[col]))
    #
    # for col in df.columns:
    #     print(col, len(Counter(df[col])), Counter(df[col]))

    # print(df.iloc[0, :])

    # seed = 2022
    # X, X_test, Y, y_test = train_test_split(X, Y, test_size=0.2, random_state=seed, shuffle=True)
    # X, X_val, Y, y_val = train_test_split(X, Y, test_size=0.2, random_state=seed, shuffle=True)
    #
    # def save_npy(x, y, type, parent_dir=f"../data/{dataset}/npy"):
    #     xcat = x.loc[:, categorical_labels].astype('object')
    #     xcont = x.loc[:, continuous_labels]
    #     y = y
    #     if type == 'train':
    #         np.save(os.path.join(parent_dir, 'X_cat_train.npy'), xcat)
    #         np.save(os.path.join(parent_dir, 'X_num_train.npy'), xcont)
    #         np.save(os.path.join(parent_dir, 'y_train.npy'), y)
    #     elif type == 'val':
    #         np.save(os.path.join(parent_dir, 'X_cat_val.npy'), xcat)
    #         np.save(os.path.join(parent_dir, 'X_num_val.npy'), xcont)
    #         np.save(os.path.join(parent_dir, 'y_val.npy'), y)
    #     elif type == 'test':
    #         np.save(os.path.join(parent_dir, 'X_cat_test.npy'), xcat)
    #         np.save(os.path.join(parent_dir, 'X_num_test.npy'), xcont)
    #         np.save(os.path.join(parent_dir, 'y_test.npy'), y)
    #
    # save_npy(X, Y, 'train')
    # save_npy(X_val, y_val, 'val')
    # save_npy(X_test, y_test, 'test')
