import numpy as np
import torch
from torch import nn

EPSILON = 1e-8


def count_leaves(depth):
    return int(2 ** depth)


def reshape_mask(mask):
    return mask.unsqueeze(1).reshape((int(mask.size(0) / 2), 2))


class Layer(nn.Module):
    def __init__(self, in_features, depth, tying_ratio):
        super().__init__()
        out_nodes = count_leaves(depth)
        mask_ones = torch.ones((out_nodes, in_features))
        mask = torch.bernoulli(mask_ones * (1 - tying_ratio)).bool()
        self.register_buffer('mask', mask)  # self.mask = mask
        self.linear = nn.Linear(in_features, out_nodes)
        self.apply_mask()

    def apply_mask(self):
        def backward_hook(grad):
            out = grad.clone()
            out[self.mask] = 0
            return out

        self.linear.weight.data[self.mask] = 0
        self.linear.weight.register_hook(backward_hook)

    def forward(self, x, path, mask):
        mask_zero = reshape_mask(mask)
        mask_one = torch.zeros_like(mask_zero)
        mask_one[mask_zero.sum(dim=1) == 1, :] = 1

        prob_right = torch.sigmoid(self.linear(x))
        prob_left = 1 - prob_right
        new_prob = torch.stack((prob_left, prob_right), dim=2)
        new_prob.masked_fill_(mask_one, 1)
        new_prob.masked_fill_(mask_zero, 0)

        new_path = path.unsqueeze(2).repeat((1, 1, 2))  # N x D x 2
        return (new_path * new_prob).view((new_path.size(0), -1))  # N x 2D

    def size(self):
        mask = self.mask.clone()
        mask[self.linear.weight.abs() < EPSILON] = 1
        return (~mask).sum(dim=1) + 1  # 1 for the bias

    def l1_loss(self):
        sum1 = self.linear.weight.abs().sum(dim=1)
        sum2 = self.linear.bias.abs()
        return sum1 + sum2

    def prune(self, ratio):
        abs_weight = torch.abs(self.linear.weight.data)
        abs_weight_np = abs_weight.cpu().numpy()
        for n in range(self.linear.weight.shape[0]):
            threshold = np.percentile(abs_weight_np[n, :], (1 - ratio) * 100)
            self.mask[n, abs_weight[n, :] < threshold] = 1
        self.apply_mask()


class CompactSDT(nn.Module):
    def __init__(self, in_features, out_classes, depth=8, tying=1.0):
        super().__init__()
        self.depth = depth
        self.logit = nn.Parameter(torch.randn(count_leaves(depth), out_classes))
        self.layers = nn.ModuleList(
            Layer(in_features, d, tying) for d in range(depth))

        # branch pruning
        for i, d in enumerate(range(1, depth + 1)):
            self.register_buffer(f'mask{i}', torch.zeros(2 ** d, dtype=torch.bool))
            self.register_buffer(f'path{i}', torch.zeros(2 ** d))

        # loss functions
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.kld_loss = torch.nn.KLDivLoss(reduction='none')
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def get_masks(self):
        return [self.__getattr__(f'mask{i}') for i in range(self.depth)]

    def get_paths(self):
        return [self.__getattr__(f'path{i}') for i in range(self.depth)]

    def propagate(self, x, accumulate=False):
        path = torch.ones((x.size(0), 1), device=x.device)
        zipped = zip(self.get_paths(), self.get_masks(), self.layers)
        for acc, mask, layer in zipped:
            path = layer(x, path, mask)
            if accumulate:
                acc += path.sum(dim=0).detach()
        return path

    def predict(self, path, mode):
        if mode == 'train':
            return self.logit.expand((path.size(0), -1, -1))
        elif mode == 'test':
            return self.logit[path.argmax(dim=1), :]
        else:
            raise ValueError(mode)

    def forward(self, x):
        path_probs = self.propagate(x, accumulate=False)
        return self.predict(path_probs, mode='test')

    def get_loss(self, x, y, accumulate=False):
        n_nodes = count_leaves(self.depth)
        path_probs = self.propagate(x, accumulate=accumulate)
        pred = self.predict(path_probs, mode='train').permute((0, 2, 1))

        if len(y.shape) == 1:
            y_ = y.unsqueeze(1).expand((-1, n_nodes))
            tq = self.ce_loss(pred, y_)
        else:
            log_pred = self.log_softmax(pred)
            y_ = y.unsqueeze(2).expand((*y.shape, n_nodes))
            tq = self.kld_loss(log_pred, y_).sum(dim=1)

        loss = (path_probs * tq).sum(dim=1).mean()
        output = self.predict(path_probs, mode='test')
        return loss, output

    def prune_nodes(self, threshold):
        rev_masks = reversed(self.get_masks())
        rev_paths = reversed(self.get_paths())

        acc_mask = None
        for mask, path in zip(rev_masks, rev_paths):
            if acc_mask is not None:
                mask[acc_mask] = 1

            path /= path.sum()
            mask[path < threshold] = 1
            path.fill_(0)
            acc_mask = reshape_mask(mask).sum(dim=1) == 2

    def prune_weights(self, ratio):
        for layer in self.layers:
            layer.prune(ratio)

    def size(self):
        def is_single_path(m):
            return reshape_mask(m).sum(dim=1) == 1

        size = self.layers[0].size().sum()
        rev_masks = reversed(self.get_masks()[:-1])
        rev_layers = reversed(self.layers[1:])

        add_mask = is_single_path(self.get_masks()[-1])
        for mask, layer in zip(rev_masks, rev_layers):
            size += layer.size()[~(mask + add_mask > 0)].sum()
            add_mask = is_single_path(mask)
        return size.item()

    def get_l1_loss(self):
        loss = self.layers[0].l1_loss().sum()
        for mask, layer in zip(self.get_masks(), self.layers[1:]):
            loss += layer.l1_loss()[~mask].sum()
        return loss

    def active_nodes(self):
        return sum((~m).sum().item() for m in self.get_masks())
