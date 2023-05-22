#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn

from mmbt.models.bert import BertEncoder,BertClf
from mmbt.models.image import ImageEncoder,ImageClf


class MultimodalLateFusionClf(nn.Module):
    def __init__(self, args):
        super(MultimodalLateFusionClf, self).__init__()
        self.args = args

        self.txtclf = BertClf(args)
        self.imgclf= ImageClf(args)


    def forward(self, txt, mask, segment, img):
        txt_out = self.txtclf(txt, mask, segment)
        img_out = self.imgclf(img)

        txt_energy = torch.log(torch.sum(torch.exp(txt_out), dim=1))
        img_energy = torch.log(torch.sum(torch.exp(img_out), dim=1))

        txt_conf = txt_energy / 10
        img_conf = img_energy / 10
        txt_conf = torch.reshape(txt_conf, (-1, 1))
        img_conf = torch.reshape(img_conf, (-1, 1))

        if self.args.df:
            txt_img_out = (txt_out * txt_conf.detach() + img_out * img_conf.detach())
        else:
            txt_conf.detach()
            img_conf.detach()
            txt_img_out=0.5*txt_out+0.5*img_out

        return txt_img_out, txt_out, img_out, txt_conf, img_conf
