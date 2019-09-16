import io
import os

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from torch import optim

import data
from models import SoftDecisionTree

DEVICE = None


def set_device():
    global DEVICE
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_valid_datasets():
    with open('../datasets.txt') as f:
        return [e.strip() for e in f.readlines()]


def evaluate(model, loader, l1reg=0) -> tuple:
    model.eval()

    sum_loss, sum_correct = 0, 0
    for x, y in loader:
        batch_size = x.size(0)
        x = x.to(DEVICE).view(batch_size, -1)
        y = y.to(DEVICE)

        loss, output = model.get_loss(x, y)
        if l1reg > 0:
            loss = loss + l1reg * model.get_l1_loss()
        pred = torch.argmax(output, dim=1)
        if len(y.shape) > 1:
            y = torch.argmax(y, dim=1)
        sum_loss += loss.item() * batch_size
        sum_correct += torch.eq(pred, y).sum().item()

    num_data = len(loader.dataset)
    loss = sum_loss / num_data
    acc = sum_correct / num_data
    return loss, acc


def train(model, loaders, l1reg, pruning, logs):
    lr = 1e-2
    epochs = 500
    val_epochs = 40
    optimizer = optim.Adam(model.parameters(), lr)

    if not os.path.exists(logs):
        with open(logs, 'w') as f:
            f.write('epoch\ttrn_loss\ttrn_acc\tval_loss\tval_acc\tparams\t'
                    'is_best\n')

    best_epoch, best_loss = -1, 1e10
    saved_model = None
    for epoch in range(epochs + 1):
        if epoch > 0:
            model.train()

            for x, y in loaders[0]:
                x = x.to(DEVICE).view(x.size(0), -1)
                y = y.to(DEVICE)

                optimizer.zero_grad()
                loss, _ = model.get_loss(x, y, accumulate=True)
                if l1reg > 0:
                    loss = loss + l1reg * model.get_l1_loss()
                loss.backward()
                optimizer.step()

        trn_loss, trn_acc = evaluate(model, loaders[0], l1reg)
        val_loss, val_acc = evaluate(model, loaders[1], l1reg)
        size = model.size()

        if pruning > 0:
            model.prune_nodes(threshold=pruning)  # 1e-6 originally

        if val_loss < best_loss:
            best_epoch = epoch
            best_loss = val_loss
            saved_model = io.BytesIO()
            torch.save(model.state_dict(), saved_model)
        elif epoch > best_epoch + val_epochs:
            break

        with open(logs, 'a') as f:
            is_best = 'BEST' if epoch == best_epoch else '-'
            f.write(f'{epoch}\t')
            f.write(f'{trn_loss:.4f}\t{trn_acc:.4f}\t')
            f.write(f'{val_loss:.4f}\t{val_acc:.4f}\t')
            f.write(f'{size}\t{is_best}\n')

    saved_model.seek(0)
    model.load_state_dict(torch.load(saved_model))


def main():
    dataset = 'abalone'
    assert dataset in get_valid_datasets()

    depth = 8
    batch_size = 128
    seed = 2019
    log_path = f'../out/edit/logs/{dataset}.txt'
    model_path = f'../out/edit/models/{dataset}.pth'
    data_path = '../data'

    distill = False  # True in the paper
    tying_ratio = 1.0  # 0.5 in the paper
    pruning_ratio = 0.5  # 0.5 in the paper
    lambda_l1reg = 0  # No optimal values
    tree_threshold = 0  # 1e-4 in paper

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if os.path.exists(log_path):
        os.remove(log_path)

    set_device()
    set_seeds(seed)

    data_dict = data.read_data(data_path, dataset, validation=True)

    model = SoftDecisionTree(
        in_features=data_dict['nx'],
        out_classes=data_dict['ny'],
        depth=depth,
        tying=tying_ratio)
    model = model.to(DEVICE)

    if distill:
        rf_path = '../out/rf/models/{}.pkl'.format(dataset)
        rf_model: RandomForestClassifier = joblib.load(rf_path)
        pred_y = rf_model.predict_proba(data_dict['trn_x'])

        trn_y = data_dict['trn_y']
        trn_y_onehot = np.zeros_like(pred_y)
        trn_y_onehot[np.arange(trn_y.shape[0]), trn_y] = 1
        data_dict['trn_y'] = (trn_y_onehot + pred_y) / 2

    trn_x = data_dict['trn_x']
    trn_y = data_dict['trn_y']
    val_x = data_dict['val_x']
    val_y = data_dict['val_y']

    trn_loader = data.to_loader(trn_x, trn_y, batch_size, shuffle=True)
    val_loader = data.to_loader(val_x, val_y, batch_size)
    loaders = (trn_loader, val_loader)

    train(model, loaders, lambda_l1reg, tree_threshold, log_path)
    if pruning_ratio < 1:
        model.prune_weights(ratio=pruning_ratio)
        train(model, loaders, lambda_l1reg, tree_threshold, log_path)

    test_x = data_dict['test_x']
    test_y = data_dict['test_y']
    test_loader = data.to_loader(test_x, test_y, batch_size)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    loss, acc = evaluate(model, test_loader, lambda_l1reg)
    print(f'Test loss: {loss:.4f}')
    print(f'Test accuracy: {acc:.4f}')
    print(f'Number of parameters {model.size()}')


if __name__ == '__main__':
    main()
