import torch
from entity_encoders.autoencoder_cat import AutoencoderWrapperCatOnly


class Encoder:

    def __init__(self, parent_dir, dataset_info,
                 encoding_type='ae_cat', decoding_type='mlp', encoding_y=False, params={},
                 seed=2023, device=torch.device('cuda')):
        self.parent_dir = parent_dir
        self.dataset_info = dataset_info

        self.encoding_type = encoding_type
        self.decoding_type = decoding_type
        self.encoding_y = encoding_y
        self.seed = seed
        self.device = device

        if self.encoding_type == 'ae_cat':
            self.wrapper = AutoencoderWrapperCatOnly(dataset_info, parent_dir, decoding=decoding_type, device=device, **params)

    def fit(self, X_train, y_train, save_path, epochs=5000, batch_size=4096, lr=1e-3, dataloader=None):
        self.wrapper.train(X_train, y_train, save_path=save_path, epochs=epochs, batch_size=batch_size, lr=lr, dataloader=dataloader)

    def eval(self, X_val, y_val, batch_size=4096):
        score = self.wrapper.eval(X_val, y_val, batch_size=batch_size)
        return score

    def encode(self, x):
        z = self.wrapper.encode(x)
        return z

    def decode(self, z, using_rec_num=False):
        if self.encoding_type == 'ae_cat':
            rec_x = self.wrapper.decode(z, using_rec_num=using_rec_num)

        return rec_x
