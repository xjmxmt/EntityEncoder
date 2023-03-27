import pandas as pd
import torch
import torch.utils.data
from torch import nn
import os
import time
import numpy as np
from math import ceil, sqrt
from sklearn import manifold
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
import pickle
import matplotlib.pyplot as plt
from entityencoder_utils.general_utils import tensor2ndarray, reconstruct_cat_features_1nn
from entity_encoders.wrapper import Wrapper


class BatchSwapNoise(nn.Module):
    """Swap Noise Module
    refer to: https://forums.fast.ai/t/porto-seguro-winning-solution-representation-learning/8499?page=2
    """

    def __init__(self, p):
        super(BatchSwapNoise, self).__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            mask = torch.rand(x.size()) > (1 - self.p)
            l1 = torch.floor(torch.rand(x.size()) * x.size(0)).type(torch.LongTensor)
            l2 = (mask.type(torch.LongTensor) * x.size(1))
            res = (l1 * l2).view(-1)
            idx = torch.arange(x.nelement()) + res
            idx[idx>=x.nelement()] = idx[idx>=x.nelement()]-x.nelement()
            return x.flatten()[idx].view(x.size())
        else:
            return x


def get_Autoencoder(n_layers, d_in, latent_dim):
    encoder_layers, decoder_layers = [], []
    encoder_layers.append(nn.Linear(d_in, latent_dim*n_layers))
    for i in range(n_layers, 1, -1):
        encoder_layers.append(nn.Linear(latent_dim*i, latent_dim*(i-1)))
        encoder_layers.append(nn.ReLU())
    for i in range(1, n_layers):
        decoder_layers.append(nn.Linear(latent_dim*i, latent_dim*(i+1)))
        decoder_layers.append(nn.ReLU())
    encoder = nn.Sequential(*encoder_layers)
    decoder = nn.Sequential(*decoder_layers)
    return encoder, decoder


class Autoencoder(nn.Module):
    def __init__(self, embedding_sizes, n_num_, latent_dim, n_layers=3, using_noise=False, p=0.1, emb_activation=None, dropout_ratio=0.6):
        super(Autoencoder, self).__init__()

        # noise module
        self.using_noise = using_noise
        if using_noise:
            self.noise = BatchSwapNoise(p)

        # categorical feature embedding
        self.embeddings = nn.ModuleList([nn.Embedding(categories, size) for categories, size in embedding_sizes])
        if emb_activation == 'sigmoid':
            self.emb_activation = nn.Sigmoid()
        elif emb_activation == 'relu':
            self.emb_activation = nn.ReLU()
        else:
            self.emb_activation = None
        self.emb_drop = nn.Dropout(dropout_ratio)

        sum_cats = sum([n_cat for n_cat, _ in embedding_sizes])
        n_emb = sum(e.embedding_dim for e in self.embeddings)
        self.n_emb, self.n_num_ = n_emb, n_num_
        d_in = n_emb + n_num_

        # self.encoder = nn.Sequential(
        #     nn.Linear(d_in, latent_dim*3),
        #     nn.ReLU(),
        #
        #     nn.Linear(latent_dim*3, latent_dim*2),
        #     nn.ReLU(),
        #
        #     nn.Linear(latent_dim*2, latent_dim),
        # )
        #
        # self.decoder = nn.Sequential(
        #     nn.Linear(latent_dim, latent_dim*2),
        #     nn.ReLU(),
        #
        #     nn.Linear(latent_dim*2, latent_dim*3),
        #     nn.ReLU(),
        # )

        self.encoder, self.decoder = get_Autoencoder(n_layers, d_in, latent_dim)

        self.decoder_cat = nn.Sequential(
            nn.Linear(latent_dim*n_layers, sum_cats),
        )

        self.decoder_num = nn.Sequential(
            nn.Linear(latent_dim*n_layers, n_num_),
        )

    def encode(self, x_cat, x_num, training=False):
        if training and self.using_noise:
            x_cat = self.noise(x_cat)
            x_num = self.noise(x_num)

        x_cat_emb = [self.embeddings[i](x_cat[:, i]) for i in range(x_cat.shape[1])]
        x_cat_emb = torch.cat(x_cat_emb, 1)
        if self.emb_activation is not None:
            x_cat_emb = self.emb_activation(x_cat_emb)
        x_cat_emb = self.emb_drop(x_cat_emb)

        x2 = x_num
        x = torch.cat([x_cat_emb, x2], 1)
        z = self.encoder(x)
        return z

    def decode(self, z):
        decoded = self.decoder(z)
        rec_cat = self.decoder_cat(decoded)
        rec_num = self.decoder_num(decoded)
        return rec_cat, rec_num

    def forward(self, x_cat, x_num, training):
        z = self.encode(x_cat, x_num, training=training)
        rec_cat, rec_num = self.decode(z)
        return rec_cat, rec_num


class CustomLoss(nn.Module):
    def __init__(self, embedding_sizes, cat_ratio=0.5, num_ratio=0.5):
        super(CustomLoss, self).__init__()
        self.embedding_sizes = embedding_sizes
        self.ce = nn.CrossEntropyLoss(reduction='mean')
        self.cat_ratio = cat_ratio
        self.mse = nn.MSELoss(reduction='mean')
        self.num_ratio = num_ratio

    def forward(self, preds, cat_targets, num_targets):
        cats, nums = preds
        total_ce, pos = 0.0, 0
        for i, (v, _) in enumerate(self.embedding_sizes):
            total_ce += self.ce(cats[:, pos:pos+v], cat_targets[:, i])
            pos += v

        cat_loss = total_ce / len(self.embedding_sizes)
        num_loss = self.mse(nums, num_targets)
        total = cat_loss * self.cat_ratio + num_loss * self.num_ratio
        return total


class Trainer:
    def __init__(self, model, loss, epochs, train_loader, n_cat, lr=1e-3, log_interval=100, device=torch.device('cuda:0')):
        self.device = device
        self.model = model.to(device)
        self.loss = loss
        self.epochs = epochs
        self.train_loader = train_loader
        self.n_cat = n_cat
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.log_interval = log_interval
        self.train_losses = []

    def _run_step(self, epoch):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(self.train_loader):
            x_cat, x_num = data[:, :self.n_cat].long().to(self.device), data[:, self.n_cat:].to(self.device)
            self.optimizer.zero_grad()
            preds = self.model(x_cat, x_num, training=True)
            loss = self.loss(preds, x_cat, x_num)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()

        if epoch % self.log_interval == 0:
            print('====> Epoch: {} Average loss: {:.8f}'.format(
                epoch, train_loss / len(self.train_loader.dataset)))
            self.train_losses.append(train_loss / len(self.train_loader.dataset))

    def run_loop(self):
        begin_time = time.time()
        self.model.to(self.device)
        self.model.train()
        print(f'log: Autoencoder training on device: {self.device}')
        for epoch in range(1, self.epochs + 1):
            self._run_step(epoch)
        print(f'log: Autoencoder training finished. Avg loss: {np.mean(self.train_losses)}')

        print(f'log: elapsed time: {time.strftime("%H:%M:%S", time.gmtime(time.time() - begin_time))}')


class AutoencoderWrapperCatOnly(Wrapper):
    def __init__(self, dataset_info: dict, parent_dir: str, latent_dim=None, n_layers=3, using_noise=False, p=0.1,
                 emb_activation=None, dropout_ratio=0.6, cat_ratio=0.5, num_ratio=0.5, decoding='mlp', device=torch.device('cuda:0')):
        """
        A wrapper for Autoencoder that mainly used for categorical features encoding

        :param parent_dir: str, log saving location
        :param ds_name: str: name of dataset
        :param device: device type (cpu or cuda:\d)
        :param real_data_path: str: .csv file path
        :param dataset_info: dict
        :param epochs: int
        :param decoding: str, ['mlp', '1nn']
        """

        assert decoding in ['mlp', '1nn']

        self.dataset_info = dataset_info
        self.parent_dir = parent_dir
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.using_noise = using_noise
        self.p = p
        self.emb_activation = emb_activation
        self.dropout_ratio = dropout_ratio
        self.cat_ratio = cat_ratio
        self.num_ratio = num_ratio if num_ratio != 0.5 else 1 - cat_ratio
        self.decoding = decoding
        self.device = device

        self.embedding_sizes = self.dataset_info['embedding_sizes']
        n_num = self.dataset_info['n_num']
        self.n_num = n_num
        self.n_cat = self.dataset_info['n_cat']
        # add y_label to n_cat
        if self.dataset_info['including_y']:
            self.n_cat += 1

        n_cat_emb = self.dataset_info['n_cat_emb']
        self.n_cat_emb = n_cat_emb

        n_num_ = n_num + int(self.dataset_info['is_regression'] and not self.dataset_info['is_y_cond'])

        d_in = n_cat_emb + n_num + int(self.dataset_info['is_regression'] and not self.dataset_info['is_y_cond'])
        self.d_in = d_in
        if self.latent_dim == 'half':
            latent_dim = d_in // 2
        elif self.latent_dim == 'one_third':
            latent_dim = d_in // 3
        elif self.latent_dim == 'sqrt':
            latent_dim = int(ceil(sqrt(d_in)))
        else:
            latent_dim = d_in
        self.latent_dim = latent_dim
        print(f'log: Autoencoder settings (d_in: {d_in}, latent_dim: {latent_dim})')

        self.model = Autoencoder(self.embedding_sizes, n_num_, latent_dim=latent_dim, n_layers=n_layers, using_noise=using_noise, p=p, emb_activation=emb_activation, dropout_ratio=dropout_ratio).to(device)
        # print('log: Autoencoder model structure: ', self.model)

        if self.decoding == '1nn':
            self.nn_classifiers = [KNeighborsClassifier(n_neighbors=1) for _ in range(self.n_cat)]

    def train(self, X_train, y_train, save_path=None, epochs=5000, batch_size=4096, lr=1e-3, dataloader=None) -> None:
        if dataloader is None:
            training_dataset = torch.utils.data.TensorDataset(X_train, y_train)
            dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

        loss = CustomLoss(self.embedding_sizes, cat_ratio=self.cat_ratio, num_ratio=self.num_ratio)
        trainer = Trainer(self.model, loss, epochs, dataloader, n_cat=self.n_cat, lr=lr, device=self.device)
        trainer.run_loop()

        if self.decoding == '1nn':
            if self.emb_activation is not None:
                for i in range(self.n_cat):
                    self.nn_classifiers[i].fit(self.model.emb_activation(self.model.embeddings[i].weight).cpu().detach().numpy(), np.arange(self.embedding_sizes[i][0]))
            else:
                for i in range(self.n_cat):
                    self.nn_classifiers[i].fit(self.model.embeddings[i].weight.cpu().detach().numpy(), np.arange(self.embedding_sizes[i][0]))

        if save_path != 'auto_searching':
            if save_path is None:
                save_path = self.parent_dir

            torch.save(self.model.state_dict(), os.path.join(save_path, "ae_weights.pt"))

            if self.decoding == '1nn':
                with open(os.path.join(save_path, '1nn_weights.pickle'), 'wb') as f:
                    pickle.dump(self.nn_classifiers, f, protocol=pickle.HIGHEST_PROTOCOL)

    def eval(self, X_val, y_val, batch_size=4096, using_rec_num=False) -> float:
        """
        :param batch_size:
        :return: reconstruction distance
        """
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        self.model.to(self.device)
        self.model.eval()
        total_distance = 0.0
        for batch_idx, (data, _) in enumerate(dataloader):
            x_cat = data[:, :self.n_cat].long().to(self.device)
            x_num = data[:, self.n_cat:].to(self.device)
            pred = self.decode(self.encode(data), using_rec_num=using_rec_num)
            pred_cat = pred[:, :self.n_cat]
            pred_num = pred[:, self.n_cat:]
            cat_distance = np.linalg.norm(tensor2ndarray(x_cat) - pred_cat, ord=1)  # Manhattan Distance
            num_distance = np.linalg.norm(tensor2ndarray(x_num) - pred_num, ord=2)
            total_distance += (cat_distance + num_distance)  # to minimize
        return total_distance

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_cat, x_num = x[:, :self.n_cat].long().to(self.device), x[:, self.n_cat:].to(self.device)
        x_cat_emb = [self.model.embeddings[i](x_cat[:, i]) for i in range(x_cat.shape[1])]
        x_cat_emb = torch.cat(x_cat_emb, 1)
        if self.emb_activation is not None:
            x_cat_emb = self.model.emb_activation(x_cat_emb)
        z = torch.hstack([x_cat_emb, x_num])
        return z

    def decode(self, z: torch.Tensor, using_rec_num=False) -> torch.Tensor:
        if self.decoding == '1nn':
            if z.shape[1] == self.n_cat_emb:
                x_cat_emb = tensor2ndarray(z[:, :self.n_cat_emb])
                rec_cat = reconstruct_cat_features_1nn(x_cat_emb, self.nn_classifiers, self.embedding_sizes)
                x = rec_cat
            else:
                x_cat_emb, x_num = tensor2ndarray(z[:, :self.n_cat_emb]), tensor2ndarray(z[:, self.n_cat_emb:])
                rec_cat = reconstruct_cat_features_1nn(x_cat_emb, self.nn_classifiers, self.embedding_sizes)
                x = np.hstack([rec_cat, x_num])

        elif self.decoding == 'mlp':
            x_num = tensor2ndarray(z[:, self.n_cat_emb:])
            z = self.model.encoder(z)
            pred_cat, rec_num = self.model.decode(z)
            i = 0
            rec_cat = []
            for n_cat, _ in self.embedding_sizes:
                tmp_cat = torch.argmax(torch.softmax(pred_cat[:, i:i+n_cat], dim=1), dim=1)
                rec_cat.append(tensor2ndarray(tmp_cat).reshape(-1, 1))
                i += n_cat
            rec_cat = np.hstack(rec_cat)
            if using_rec_num:
                x = np.hstack([rec_cat, tensor2ndarray(rec_num)])
            else:
                x = np.hstack([rec_cat, x_num])

        else:
            raise "Wrong type of method."

        return x

    # def visualize(self):
    #     self.model.load_state_dict(torch.load(os.path.join(self.parent_dir, f"ae_weights.pt"), map_location=self.device))
    #     unnorm_embs = self.model.embeddings
    #
    #     # with open(os.path.join(self.parent_dir, 'ae_normalizer.pickle'), 'rb') as f:
    #     #     norm_embd, _ = pickle.load(f)
    #     norm_embd = []
    #
    #     label_encoder = self.dataset_info['cat_encoder']
    #     categories = label_encoder.categories_
    #     print(categories[0], categories[1])
    #
    #     col = 1
    #     unnorm_embs = unnorm_embs[col].weight.cpu().detach().numpy()
    #     tsne = manifold.TSNE(n_components=2, init='pca', random_state=2023)
    #     emb_tsne = tsne.fit_transform(unnorm_embs)
    #
    #     df = pd.read_csv(f'../data/{self.ds_name}/processed_train.csv')
    #     # protocol_type = df['protocol_type']
    #     # service = df['service']
    #     # udp_service = df[df['protocol_type'].str.match('udp')]['service']
    #     # tcp_service = df[df['protocol_type'].str.match('tcp')]['service']
    #     # icmp_service = df[df['protocol_type'].str.match('icmp')]['service']
    #
    #     print(len(Counter(df['service'])))
    #
    #     def func(protocol):
    #         service_df = df[df['protocol_type'].str.match(protocol)]['service']
    #         service_dict = Counter(service_df)
    #         all_service_df = df['service']
    #         all_service_dict = Counter(all_service_df)
    #         res = []
    #         for k, v in service_dict.items():
    #             if v == all_service_dict[k]:
    #                 res.append(k)
    #         return res
    #
    #     udp_service = func('udp')
    #     tcp_service = func('tcp')
    #     icmp_service = func('icmp')
    #
    #     color_dict = {"udp": "r", "tcp": "g", "icmp": "b", "mixed": "k"}
    #
    #     x_min, x_max = emb_tsne.min(0), emb_tsne.max(0)
    #     X_norm = (emb_tsne - x_min) / (x_max - x_min)  # standardize
    #     plt.figure(figsize=(8, 8))
    #     scatter_dict = {"udp": [], "tcp": [], "icmp": [], "mixed": []}
    #     for idx, cat in enumerate(categories[col]):
    #         if cat in udp_service: scatter_dict["udp"].append(idx)
    #         elif cat in tcp_service: scatter_dict["tcp"].append(idx)
    #         elif cat in icmp_service: scatter_dict["icmp"].append(idx)
    #         else: scatter_dict["mixed"].append(idx)
    #     for k, v in scatter_dict.items():
    #         plt.scatter(X_norm[v, 0], X_norm[v, 1], c=color_dict[k], label=k)
    #     # plt.scatter(X_norm[:, 0], X_norm[:, 1])
    #     for k in range(len(X_norm)):
    #         plt.annotate(categories[col][k], xy=(X_norm[k, 0], X_norm[k, 1]),
    #                      xytext=(X_norm[k, 0] + 0.01, X_norm[k, 1] + 0.01))
    #     plt.title(f"t-SNE on protocol_type & service")
    #     plt.legend()
    #     plt.savefig(os.path.join(self.parent_dir, "tsne_prt_srv.png"), dpi=300)
    #     plt.show()
    #
    #     # Note all in one:
    #     # if len(norm_embd) == 0:
    #     #     fig, axes = plt.subplots(len(unnorm_embs), 1)
    #     # else:
    #     #     fig, axes = plt.subplots(len(unnorm_embs), 2)
    #     # ax_list = []
    #     # for i in range(2):
    #     #     if i == 0:
    #     #         # plots for unnormed embeddings
    #     #         for j in range(len(unnorm_embs)):
    #     #             if len(norm_embd) == 0:
    #     #                 ax = axes[j]
    #     #             else:
    #     #                 ax = axes[j][i]
    #     #             ax.autoscale(True)
    #     #             tmpembs = unnorm_embs[j].weight.cpu().detach().numpy()
    #     #             if tmpembs.shape[1] == 1:
    #     #                 ax.scatter(tmpembs[:, 0], np.zeros(tmpembs.shape)[:, 0])
    #     #                 ax_list.append(ax)
    #     #                 for k in range(len(tmpembs)):
    #     #                     ax.annotate(categories[j][k], xy=(tmpembs[:, 0][k], 0), xytext=(tmpembs[:, 0][k] + 0.01, 0 + 0.01))
    #     #             elif tmpembs.shape[1] == 2:
    #     #                 ax.scatter(tmpembs[:, 0], tmpembs[:, 1])
    #     #                 ax_list.append(ax)
    #     #                 for k in range(len(tmpembs)):
    #     #                     ax.annotate(categories[j][k], xy=(tmpembs[:, 0][k], tmpembs[:, 1][k]), xytext=(tmpembs[:, 0][k] + 0.01, tmpembs[:, 1][k] + 0.01))
    #     #             elif tmpembs.shape[1] == 3:
    #     #                 if len(norm_embd) == 0:
    #     #                     axes[j].remove()
    #     #                 else:
    #     #                     axes[j][i].remove()
    #     #                 ax = fig.add_subplot(len(unnorm_embs), 2, 2 * j + i + 1, projection='3d')
    #     #                 ax.scatter(tmpembs[:, 0], tmpembs[:, 1], tmpembs[:, 2])
    #     #                 ax_list.append(ax)
    #     #                 for k in range(len(tmpembs)):
    #     #                     ax.text(tmpembs[:, 0][k], tmpembs[:, 1][k], tmpembs[:, 2][k], s=categories[j][k],
    #     #                             zorder=1, color='k')
    #     #             else:
    #     #                 if len(norm_embd) == 0:
    #     #                     axes[j].remove()
    #     #                 else:
    #     #                     axes[j][i].remove()
    #     #                 ax = fig.add_subplot(len(unnorm_embs), 2, 2 * j + i + 1, projection='3d')
    #     #                 im = ax.scatter(tmpembs[:, 0], tmpembs[:, 1], tmpembs[:, 2], c=tmpembs[:, 3],
    #     #                                 cmap=plt.hot())
    #     #                 ax_list.append(ax)
    #     #                 plt.colorbar(im, ax=ax)
    #     #     else:
    #     #         # plots for normed embeddings
    #     #         for j in range(len(norm_embd)):
    #     #             ax = axes[j][i]
    #     #             ax.autoscale(True)
    #     #             tmpembs = norm_embd[j]
    #     #             if tmpembs.shape[1] == 1:
    #     #                 ax.sharex=ax_list[j]
    #     #                 ax.scatter(tmpembs[:, 0], np.zeros(tmpembs.shape)[:, 0])
    #     #                 for k in range(len(tmpembs)):
    #     #                     ax.annotate(categories[j][k], xy=(tmpembs[:, 0][k], 0), xytext=(tmpembs[:, 0][k] + 0.1, 0 + 0.1))
    #     #             elif tmpembs.shape[1] == 2:
    #     #                 ax.sharex = ax_list[j]
    #     #                 ax.scatter(tmpembs[:, 0], tmpembs[:, 1])
    #     #                 for k in range(len(tmpembs)):
    #     #                     ax.annotate(categories[j][k], xy=(tmpembs[:, 0][k], tmpembs[:, 1][k]), xytext=(tmpembs[:, 0][k] + 0.1, tmpembs[:, 1][k] + 0.1))
    #     #             elif tmpembs.shape[1] == 3:
    #     #                 axes[j][i].remove()
    #     #                 ax = fig.add_subplot(len(norm_embd), 2, 2 * j + i + 1, projection='3d')
    #     #                 ax.scatter(tmpembs[:, 0], tmpembs[:, 1], tmpembs[:, 2])
    #     #                 ax_list.append(ax)
    #     #                 for k in range(len(tmpembs)):
    #     #                     ax.text(tmpembs[:, 0][k], tmpembs[:, 1][k], tmpembs[:, 2][k], s=categories[j][k],
    #     #                             zorder=1, color='k')
    #     #             else:
    #     #                 axes[j][i].remove()
    #     #                 ax = fig.add_subplot(len(norm_embd), 2, 2 * j + i + 1, projection='3d')
    #     #                 im = ax.scatter(tmpembs[:, 0], tmpembs[:, 1], tmpembs[:, 2], c=tmpembs[:, 3],
    #     #                                 cmap=plt.hot())
    #     #                 ax_list.append(ax)
    #     #                 plt.colorbar(im, ax=ax)
    #     # plt.show()

    # def embedding_analyze(self):
    #     self.model.load_state_dict(torch.load(os.path.join(self.parent_dir, f"ae_weights.pt"), map_location=self.device))
    #     unnorm_embs = self.model.embeddings
    #
    #     with open(os.path.join(self.parent_dir, 'ae_normalizer.pickle'), 'rb') as f:
    #         norm_embd, _ = pickle.load(f)
    #
    #     print('hola')
    #     print(len(unnorm_embs))
    #     print(len(norm_embd))
    #
    #     label_encoder = self.dataset_info['cat_encoder']
    #     categories = label_encoder.categories_
    #     print(categories)
    #
    #     color_dict = {s: i for i, s in enumerate(categories[1])}
    #     inverse_color_dict = {i: s for i, s in enumerate(categories[1])}
    #
    #     X = norm_embd[1]
    #     choc_embedding = X[color_dict['Chocolate']]
    #     choc_point_embedding = X[color_dict['Chocolate Point']]
    #     tortie_embedding = X[color_dict['Silver']]
    #
    #     new_embedding = choc_point_embedding - choc_embedding + tortie_embedding  #
    #     print(choc_point_embedding - choc_embedding)
    #
    #     # protocol_dict = {s: i for i, s in enumerate(categories[1])}
    #     # service_dict = {s: i for i, s in enumerate(categories[2])}
    #     # inverse_service_dict = {i: s for i, s in enumerate(categories[2])}
    #     #
    #     # # tcp_embedding = norm_embd[1][protocol_dict['tcp']]
    #     # # icmp_embedding = norm_embd[1][protocol_dict['icmp']]
    #     # # udp_embedding = norm_embd[1][protocol_dict['udp']]
    #     # http_embedding = norm_embd[2][service_dict['http']]
    #     # ftp_embedding = norm_embd[2][service_dict['ftp']]
    #     # telnet_embedding = norm_embd[2][service_dict['telnet']]
    #     #
    #     # ntp_u_embedding = norm_embd[2][service_dict['ntp_u']]
    #     # tftp_u_embedding = norm_embd[2][service_dict['tftp_u']]
    #     # domain_u_embedding = norm_embd[2][service_dict['domain_u']]
    #     #
    #     from scipy.spatial import distance
    #     # # base_cosine = distance.cosine(udp_embedding.reshape(1, -1), ntp_u_embedding.reshape(1, -1))
    #     # # print(base_cosine)
    #     #
    #     # # new_embedding = http_embedding - ftp_embedding + ntp_u_embedding  # domain_u
    #     # # new_embedding = http_embedding - ftp_embedding + tftp_u_embedding  # domain_u
    #     # # new_embedding = http_embedding - ftp_embedding + domain_u_embedding  # domain_u
    #     # new_embedding = domain_u_embedding
    #     #
    #     from sample import euclidean_dist
    #
    #     res_dist = float('inf')
    #     res_idx = -1
    #     for idx, emb in enumerate(norm_embd[1]):
    #         # tmp_dist = euclidean_dist(emb.reshape(1, -1), new_embedding.reshape(1, -1))
    #         tmp_dist = distance.cosine(emb.reshape(1, -1), new_embedding.reshape(1, -1))
    #         if tmp_dist < res_dist:
    #             print(tmp_dist, res_dist)
    #             res_dist = tmp_dist
    #             res_idx = idx
    #     print(res_idx, inverse_color_dict[res_idx])
    #     #
    #     # # from sklearn.cluster import KMeans
    #     # # X = norm_embd[2]
    #     # # y_pred = KMeans(n_clusters=2, random_state=9).fit_predict(X)
    #
    #
    #     # from sklearn.cluster import DBSCAN
    #     # X = norm_embd[1]
    #     # y_pred = DBSCAN().fit_predict(X)
    #
    #     # fig = plt.figure()
    #     # ax = fig.add_subplot(111, projection='3d')
    #     # ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_pred)
    #     #
    #     # for k in range(len(X)):
    #     #     ax.text(X[:, 0][k], X[:, 1][k], X[:, 2][k], s=categories[1][k],
    #     #             zorder=1, color='k')
    #     # plt.show()
    #
    #     # print(y_pred)
    #     #
    #     # i = 0
    #     # l1, l2 = [], []
    #     # for _, v in inverse_color_dict.items():
    #     #     if y_pred[i] == -1:
    #     #         l1.append(v)
    #     #     else:
    #     #         l2.append(v)
    #     #     i += 1
    #     # print(l1)
    #     # print(l2)


if __name__ == '__main__':
    ae_wrapper = AutoencoderWrapperCatOnly("../test_cases/exp/adult/99", "adult", real_data_path="../test_cases/data/adult/train.csv", epochs=5000)
    # ae_wrapper.eval(batch_size=8)
    # ae_wrapper.train(save_path="../exp/tempdir/cover/ae")
    # ae_wrapper.visualize()
    # ae_wrapper.embedding_analyze()