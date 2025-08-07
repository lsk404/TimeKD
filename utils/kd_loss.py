import torch.nn as nn
from .similar_utils import *
from copy import deepcopy
from .losses import mape_loss, mase_loss, smape_loss


loss_dict = {
    "l1": nn.L1Loss(),
    "smooth_l1": nn.SmoothL1Loss(),
    "ce": nn.CrossEntropyLoss(),
    "mse": nn.MSELoss(),
    "smape": smape_loss(),
    "mape": mape_loss(),
    "mase": mase_loss(),
}


class KDLoss(nn.Module):
    def __init__(self, feature_loss, fcst_loss, recon_loss, att_loss, feature_w=0.01, fcst_w=1.0, recon_w = 0.5, att_w = 0.01):
        super(KDLoss, self).__init__()
        self.fcst_w = fcst_w
        self.feature_w = feature_w
        self.recon_w = recon_w
        self.att_w = att_w

        self.feature_loss = loss_dict[feature_loss]
        self.fcst_loss = loss_dict[fcst_loss]
        self.recon_loss = loss_dict[recon_loss]
        self.att_loss = loss_dict[att_loss]

    def forward(self, ts_enc, prompt_enc, ts_out, prompt_out, ts_att_last, prompt_att_last, real):
    
        feature_loss = self.feature_loss(ts_enc, prompt_enc)     
        fcst_loss = self.fcst_loss(ts_out, real)
        recon_loss = self.recon_loss(prompt_out, real)
        att_loss = self.att_loss(ts_att_last, prompt_att_last)

        total_loss = self.fcst_w * fcst_loss + self.feature_w * feature_loss + self.recon_w * recon_loss + self.att_w * att_loss

        return total_loss