import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class PretrainModel(nn.Module):
    def __init__(self, embedding_sizes, n_cont, n_classes):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(categories, size) for categories, size in embedding_sizes])
        n_emb = sum(e.embedding_dim for e in self.embeddings)  # length of all embeddings combined
        self.n_emb, self.n_cont = n_emb, n_cont
        self.lin1 = nn.Linear(self.n_emb + self.n_cont, 200)
        self.lin2 = nn.Linear(200, 70)
        self.lin3 = nn.Linear(70, n_classes)
        self.bn1 = nn.BatchNorm1d(self.n_cont)
        self.bn2 = nn.BatchNorm1d(200)
        self.bn3 = nn.BatchNorm1d(70)
        self.emb_drop = nn.Dropout(0.6)
        self.drops = nn.Dropout(0.3)

    def forward(self, x_cat, x_cont):
        x = [e(x_cat[:, i]) for i, e in enumerate(self.embeddings)]
        x = torch.cat(x, 1)
        x = self.emb_drop(x)
        x2 = self.bn1(x_cont)
        x = torch.cat([x, x2], 1)
        x = F.relu(self.lin1(x))
        x = self.drops(x)
        x = self.bn2(x)
        x = F.relu(self.lin2(x))
        x = self.drops(x)
        x = self.bn3(x)
        x = self.lin3(x)
        return x


class MixedDataset(Dataset):
    def __init__(self, X_cat, X_cont, Y):
        self.X1 = X_cat  # categorical columns
        self.X2 = X_cont  # numerical columns
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X1[idx], self.X2[idx], self.y[idx]


def get_optimizer(model, lr=0.001, wd=0.0):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optim = torch.optim.Adam(parameters, lr=lr, weight_decay=wd)
    return optim


def train_model(model, optim, train_dl, device, loss_type='ce'):
    model.train()
    total = 0
    sum_loss = 0
    for x1, x2, y in train_dl:
        x1 = x1.to(device)
        x2 = x2.to(device)
        batch = x1.shape[0]
        output = model(x1, x2)
        if loss_type == 'ce':
            y = y.to(device)
            loss = F.cross_entropy(output, y, reduction='sum')
        else:
            y = y.float().to(device)
            loss = F.mse_loss(output, y.reshape(-1, 1), reduction='sum')
        optim.zero_grad()
        loss.backward()
        optim.step()
        total += batch
        sum_loss += loss.item()
    return sum_loss / total


def val_loss(model, valid_dl, device, loss_type='ce'):
    model.eval()
    total = 0
    sum_loss = 0
    correct = 0
    for x1, x2, y in valid_dl:
        x1 = x1.to(device)
        x2 = x2.to(device)
        current_batch_size = x1.shape[0]
        output = model(x1, x2)
        if loss_type == 'ce':
            y = y.to(device)
            loss = F.cross_entropy(output, y, reduction='sum')
            pred = torch.max(output, 1)[1]
            correct += (pred == y).float().sum().item()
        else:
            y = y.float().to(device)
            loss = F.mse_loss(output, y.reshape(-1, 1), reduction='sum')
        sum_loss += loss.item()
        total += current_batch_size
    if loss_type == 'ce':
        print("valid loss %.3f and accuracy %.3f" % (sum_loss / total, correct / total))
        return sum_loss / total, correct / total
    else:
        print("valid loss %.3f" % (sum_loss / total))
        return sum_loss / total


def train_loop(model, train_dl, valid_dl, epochs, lr=0.01, wd=0.0, is_regression=False, device=torch.device('cuda')):
    optim = get_optimizer(model, lr=lr, wd=wd)
    if is_regression: loss_type = 'mse'
    else: loss_type = 'ce'
    for i in range(epochs):
        loss = train_model(model, optim, train_dl, device, loss_type=loss_type)
        print("training loss: ", loss)
        val_loss(model, valid_dl, device, loss_type=loss_type)


def pretrain(dataset_name, trainset, y_train, valset, y_val, embedding_sizes, parent_dir, n_classes, is_regression=False, batch_size=1000, device=torch.device('cuda:0')):
    """
    Train a task specified model to get entity embeddings.

    :param dataset_name:
    :param trainset:
    :param y_train:
    :param valset:
    :param y_val:
    :param embedding_sizes:
    :param parent_dir:
    :param batch_size:
    :param device:
    :return:
    """
    n_cat = len(embedding_sizes)
    n_cont = trainset.shape[1] - n_cat

    X_train_cat = trainset[:, :n_cat]
    X_train_cont = trainset[:, n_cat:]
    X_val_cat = valset[:, :n_cat]
    X_val_cont = valset[:, n_cat:]

    train_ds = MixedDataset(X_train_cat.int().to(device), X_train_cont, y_train)
    valid_ds = MixedDataset(X_val_cat.int().to(device), X_val_cont, y_val)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=True)

    model = PretrainModel(embedding_sizes, n_cont, n_classes)
    model.to(device)

    train_loop(model, train_dl, valid_dl, epochs=20, lr=0.05, wd=0.00001, is_regression=is_regression, device=device)

    entity_embeddings = model.embeddings
    torch.save(entity_embeddings.state_dict(), os.path.join(parent_dir, 'embedding.pt'))

    return entity_embeddings
