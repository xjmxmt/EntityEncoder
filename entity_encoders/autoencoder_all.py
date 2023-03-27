import torch
import torch.utils.data
from sklearn.neighbors import KNeighborsClassifier
from torch import nn
from math import ceil, sqrt
import os
import time
import numpy as np
from entityencoder_utils.general_utils import reconstruct_cat_features_1nn
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


class Autoencoder(nn.Module):
    def __init__(self, embedding_sizes, n_num_, latent_dim, p=0.1):
        super(Autoencoder, self).__init__()

        # noise module
        # self.noise = BatchSwapNoise(p)

        # categorical feature embedding
        self.embeddings = nn.ModuleList([nn.Embedding(categories, size) for categories, size in embedding_sizes])
        self.emb_activation = nn.Sigmoid()
        self.emb_drop = nn.Dropout(0.6)

        sum_cats = sum([n_cat for n_cat, _ in embedding_sizes])
        n_emb = sum(e.embedding_dim for e in self.embeddings)  # length of all embeddings combined
        self.n_emb, self.n_num_ = n_emb, n_num_
        d_in = n_emb + n_num_

        # self.bn1 = nn.BatchNorm1d(self.n_cont_)

        self.encoder = nn.Sequential(
            nn.Linear(d_in, latent_dim*3),
            nn.ReLU(True),
            # nn.BatchNorm1d(num_features=latent_dim*3),

            nn.Linear(latent_dim*3, latent_dim*2),
            nn.ReLU(True),
            # nn.BatchNorm1d(num_features=latent_dim*2),

            nn.Linear(latent_dim*2, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim*2),
            nn.ReLU(True),
            # nn.Dropout(0.1),
            # nn.BatchNorm1d(num_features=latent_dim*2),

            nn.Linear(latent_dim*2, latent_dim*3),
            nn.ReLU(True),
            # nn.Dropout(0.1),
            # nn.BatchNorm1d(num_features=latent_dim*3),
        )

        self.decoder_cat = nn.Sequential(
            nn.Linear(latent_dim*3, sum_cats),
            # nn.Dropout(0.1)
        )

        self.decoder_cont = nn.Sequential(
            nn.Linear(latent_dim*3, n_num_),
            # nn.Dropout(0.1)
        )

    def encode(self, x_cat, x_num, training=False):
        # if training:
        #     x_cat = self.noise(x_cat)
        #     x_cont = self.noise(x_cont)
        # x_cat_emb = [e(x_cat[:, i]) for i, e in enumerate(self.embeddings)]

        x_cat_emb = [self.embeddings[i](x_cat[:, i]) for i in range(x_cat.shape[1])]
        x_cat_emb = torch.cat(x_cat_emb, 1)
        x_cat_emb = self.emb_activation(x_cat_emb)
        x_cat_emb = self.emb_drop(x_cat_emb)

        # x2 = self.bn1(x_cont)
        x2 = x_num
        x = torch.cat([x_cat_emb, x2], 1)
        z = self.encoder(x)
        return z

    def decode(self, z):
        decoded = self.decoder(z)
        rec_cat = self.decoder_cat(decoded)
        rec_num = self.decoder_cont(decoded)
        return rec_cat, rec_num

    def forward(self, x_cat, x_num, training):
        z = self.encode(x_cat, x_num, training=training)
        rec_cat, rec_num = self.decode(z)
        return rec_cat, rec_num


class CustomLoss(nn.Module):
    def __init__(self, embedding_sizes):
        super(CustomLoss, self).__init__()
        self.embedding_sizes = embedding_sizes
        self.ce = nn.CrossEntropyLoss(reduction='mean')
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, preds, cat_targets, num_targets):
        cats, nums = preds
        total_ce, pos = 0.0, 0
        for i, (v, _) in enumerate(self.embedding_sizes):
            total_ce += self.ce(cats[:, pos:pos+v], cat_targets[:, i])
            pos += v

        cat_loss = total_ce / len(self.embedding_sizes)
        cont_loss = self.mse(nums, num_targets)
        total = cat_loss + cont_loss
        return total


class Trainer:
    def __init__(self, model, loss, epochs, train_loader, n_cat, lr=1e-3, device=torch.device('cuda:0'), log_interval=100):
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
            print('====> Epoch: {} Average loss: {:.4f}'.format(
                epoch, train_loss / len(self.train_loader.dataset)))
            self.train_losses.append(train_loss / len(self.train_loader.dataset))

    def run_loop(self, save_path=None):
        begin_time = time.time()
        self.model.train()
        for epoch in range(1, self.epochs + 1):
            self._run_step(epoch)
        print(f'log: Autoencoder training finished. Avg loss: {np.mean(self.train_losses)}')
        if save_path is not None:
            torch.save(self.model.state_dict(), os.path.join(save_path, "ae_weights.pt"))
        print(f'log: elapsed time: {time.strftime("%H:%M:%S", time.gmtime(time.time() - begin_time))}')


class AutoencoderWrapperBoth(Wrapper):
    def __init__(self, dataset_info: dict, parent_dir: str, decoding='mlp', device=torch.device('cuda:0')):
        """
        A wrapper for Autoencoder that encodes both categorical and numerical features

        :param parent_dir: str, log saving location
        :param ds_name: str: name of dataset
        :param device: device type (cpu or cuda:\d)
        :param real_data_path: str: .csv file path
        :param dataset_info: dict
        :param epochs: int
        :param decoding: str, ['mlp', '1nn']
        """

        self.dataset_info = dataset_info
        self.parent_dir = parent_dir
        self.decoding = decoding
        self.device = device

        self.embedding_sizes = self.dataset_info['embedding_sizes']
        n_num = self.dataset_info['n_num']
        self.n_num = n_num
        self.n_cat = self.dataset_info['n_cat']
        n_cat_emb = self.dataset_info['n_cat_emb']
        self.n_cat_emb = n_cat_emb

        n_num_ = n_num + int(self.dataset_info['is_regression'] and not self.dataset_info['is_y_cond'])

        d_in = n_cat_emb + n_num + int(self.dataset_info['is_regression'] and not self.dataset_info['is_y_cond'])
        self.d_in = d_in
        # latent_dim = int(ceil(sqrt(d_in)))
        # latent_dim = d_in // 2
        latent_dim = d_in // 3
        self.latent_dim = latent_dim
        print(f'log: Autoencoder settings (d_in: {d_in}, latent_dim: {latent_dim})')

        self.model = Autoencoder(self.embedding_sizes, n_num_, latent_dim=latent_dim).to(device)

        if self.decoding == '1nn':
            self.nn_classifiers = [KNeighborsClassifier(n_neighbors=1) for _ in range(self.n_cat)]

    def train(self, X_train, y_train, save_path=None, epochs=5000, batch_size=4096, lr=1e-3):
        training_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

        loss = CustomLoss(self.embedding_sizes)
        trainer = Trainer(self.model, loss, epochs, dataloader, n_cat=self.n_cat, lr=lr, device=self.device)
        trainer.run_loop(save_path=save_path)

        if self.decoding == '1nn':
            for i in range(self.n_cat):
                self.nn_classifiers[i].fit(self.model.emb_activation(self.model.embeddings[i].weight).cpu().detach().numpy(),
                                           np.arange(self.embedding_sizes[i][0]))

    def eval(self, X_val, y_val, batch_size=4096):
        evaluating_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        dataloader = torch.utils.data.DataLoader(evaluating_dataset, batch_size=batch_size, shuffle=True)
        self.model.to(self.device)
        self.model.eval()
        total_score = 0.0
        for batch_idx, (data, _) in enumerate(dataloader):
            x_cat = data[:, :self.n_cat].long().to(self.device)
            x_num = data[:, self.n_cat:].to(self.device)
            # print(f'log: val data: {x_cat}')
            pred_cat, pred_num = self.decode(self.encode(data))
            # print(f'log: categorical data after AE: {pred_cat}')
            # print(f'log: numerical data after AE: {pred_num}')

            cat_distance = np.linalg.norm(x_cat - pred_cat, ord=1)  # Manhattan Distance
            num_distance = np.sqrt(np.linalg.norm(x_num - pred_num, ord=2))  # Euclidean Distance
            total_score += (cat_distance + num_distance)  # to minimize
        return total_score

    def encode(self, x: torch.Tensor):
        x_cat, x_num = x[:, :self.n_cat_emb].long().to(self.device), x[:, self.n_cat_emb:].to(self.device)
        z = self.model.encode(x_cat, x_num, training=False)
        return z

    def decode(self, z:torch.Tensor):
        pred_cat, rec_num = self.model.decode(z)

        if self.decoding == '1nn':
            rec_cat = reconstruct_cat_features_1nn(pred_cat, self.nn_classifiers, self.embedding_sizes)
        elif self.decoding == 'mlp':
            i = 0
            rec_cat = []
            for n_cat, _ in self.embedding_sizes:
                tmp_cat = torch.argmax(torch.softmax(pred_cat[:, i:i+n_cat], dim=1), dim=1)
                rec_cat.append(tmp_cat.numpy().reshape(-1, 1))
                i += n_cat
            rec_cat = np.hstack(rec_cat)
        else:
            raise "Wrong type of method."

        x = np.hstack([rec_cat, rec_num])

        return x


if __name__ == '__main__':
    ae_wrapper = AutoencoderWrapperBoth("../exp/adult/ae-whole-both", 'adult', real_data_path="../data/adult/train.csv")
    # ae_wrapper.train("../exp/adult/ae-whole-both")
    # ae_wrapper.eval()
