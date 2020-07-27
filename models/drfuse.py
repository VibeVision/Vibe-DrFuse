import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50, ResNet50_Weights

from .ehr_transformer import EHRTransformer

class DrFuseModel(nn.Module):
    def __init__(self, hidden_size, num_classes, ehr_dropout, ehr_n_layers, ehr_n_head,
                 cxr_model='swin_s', logit_average=False):
        super().__init__()
        self.num_classes = num_classes
        self.logit_average = logit_average
        self.ehr_model = EHRTransformer(input_size=76, num_classes=num_classes,
                                        d_model=hidden_size, n_head=ehr_n_head,
                                        n_layers_feat=1, n_layers_shared=ehr_n_layers,
                                        n_layers_distinct=ehr_n_layers,
                             