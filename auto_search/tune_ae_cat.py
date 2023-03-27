import optuna
from entity_encoders.autoencoder_cat import AutoencoderWrapperCatOnly
from auto_search.rtdl_wrapper import rtdl_evaluation


def objective(trial, raw_data, ds_info, parent_dir, metrics='ml_efficiency'):
    assert metrics in ['distance', 'ml_efficiency', 'mixed']
    X_train, y_train = raw_data['X_train'], raw_data['y_train']
    X_val, y_val = raw_data['X_val'], raw_data['y_val']

    # Todo: random forest to visualize
    lr = trial.suggest_loguniform('lr', 0.00001, 0.001)
    epochs = trial.suggest_categorical('epochs', [500, 1000, 2000])
    latent_dim = trial.suggest_categorical('latent_dim', [None, 'half', 'one_third', 'sqrt'])
    n_layers = trial.suggest_categorical('n_layers', [2, 3, 5])
    # using_noise = trial.suggest_categorical('using_noise', [True, False])
    # emb_activation = trial.suggest_categorical('emb_activation', [None, 'sigmoid', 'relu'])
    cat_ratio = trial.suggest_categorical('cat_ratio', [0.5, ds_info['n_cat']/(ds_info['n_cat']+ds_info['n_num'])])

    ae_wrapper = AutoencoderWrapperCatOnly(ds_info, parent_dir, latent_dim=latent_dim, n_layers=n_layers,
                                           cat_ratio=cat_ratio, num_ratio=1-cat_ratio)
    ae_wrapper.train(X_train, y_train, save_path='auto_searching', epochs=epochs, lr=lr)
    ae_wrapper.model.eval()

    d_out = 1 if ds_info['is_regression'] or ds_info['n_classes'] == 2 else ds_info['n_classes']

    eval_score = 0.0
    if metrics == 'distance':
        distance = ae_wrapper.eval(X_val, y_val)
        eval_score = -distance
    elif metrics == 'ml_efficiency':
        score = rtdl_evaluation(raw_data, d_in=ae_wrapper.d_in, d_out=d_out,
                                ds_info=ds_info, encoder_wrapper=ae_wrapper)
        eval_score = score
    elif metrics == 'mixed':
        distance = ae_wrapper.eval(X_val, y_val)
        score = rtdl_evaluation(raw_data, d_in=ae_wrapper.d_in, d_out=d_out,
                                ds_info=ds_info, encoder_wrapper=ae_wrapper)
        eval_score = (score - distance) / 2

    print(f'log: metrics tyep: {metrics}, value: {eval_score}')

    return eval_score


def get_best_hyperparams(objective, raw_data, ds_info, parent_dir, metrics='ml_efficiency'):
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=2023)
    )

    func = lambda trial: objective(trial, raw_data, ds_info, parent_dir, metrics=metrics)

    study.optimize(func, n_trials=20, show_progress_bar=True)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    return trial