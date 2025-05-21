import torch
from torch import nn


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2, weights=None, device=None):
        self.weights = weights
        learnable = True
        if weights is not None:
            learnable = False
        self.learnable = learnable
        super(AutomaticWeightedLoss, self).__init__()
        if learnable:
            params = torch.ones(num, requires_grad=learnable)
            self.params = torch.nn.Parameter(params)
        else:
            self.params = torch.tensor(weights, requires_grad=learnable, device=device)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


class LossFunc_Difficulty_SQUARE(nn.Module):
    def __init__(self):
        super().__init__()
        self.auto_weighted_loss = AutomaticWeightedLoss(3)
        self.mse_func = torch.nn.MSELoss()
        self.bce_func = nn.BCELoss()
        self.info_auto_weighted_loss = AutomaticWeightedLoss(2)

    def forward(self, dec_pred, cm_readhead, atten_cmm_readhead, y, error_pred, ensemble_pred, item_size, support_info,
                support_y):
        print('not implemented')
        exit(0)
        return None

    def batch_forward_dec(self, batch_dec_pred, batch_cm_readhead, batch_y, item_size_list, zipf_info, stream_info1,
                          stream_info2):
        # compute dec loss
        start_pos = 0
        epsilon = 0.001
        batch_loss = 0
        for item_size in item_size_list:
            dec_pred = batch_dec_pred[start_pos:start_pos + item_size]
            cm_readhead = batch_cm_readhead[start_pos:start_pos + item_size]
            y = batch_y[start_pos:start_pos + item_size]
            start_pos += item_size

            dec_are = ((dec_pred - y) / y).abs().mean()
            dec_aae = (dec_pred - y).abs().mean()
            dec_mse = (dec_pred - y).square().mean()
            # dec_length =  torch.abs(dec_pred.sum()-y.sum())/y.sum()
            with torch.no_grad():
                cm_are = ((cm_readhead - y) / y).abs().mean().detach()
                cm_aae = (cm_readhead - y).abs().mean().detach()
                cm_mse = (cm_readhead - y).square().mean().detach()
            importance_dec_loss = self.auto_weighted_loss((dec_are / (cm_are + epsilon)).square(),
                                                          (dec_aae / (cm_aae + epsilon)).square(),
                                                          (dec_mse / (cm_mse + epsilon)).square())
            batch_loss += importance_dec_loss / len(item_size_list)
        zipf_loss = (zipf_info - stream_info1).square().mean()
        item_size_tensor = torch.tensor(item_size_list, device=zipf_loss.device).view(-1, 1).float()
        item_size_loss = (stream_info2 - item_size_tensor).square().mean()
        zipf_and_item_loss = self.info_auto_weighted_loss(zipf_loss, item_size_loss)
        return batch_loss + 0.1 * zipf_and_item_loss

    def forward_dec(self, dec_pred, cm_readhead, y, zipf_info, stream_info1, stream_info2):
        item_size_list = [dec_pred.shape[0]]
        return self.batch_forward_dec(dec_pred, cm_readhead, y, item_size_list, zipf_info.view(-1, 1), stream_info1,
                                      stream_info2)
