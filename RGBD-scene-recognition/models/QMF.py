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
from models.image import ImageEncoder
import torch.nn.functional as F



class QMF(nn.Module):
    def __init__(self, args):
        super(QMF, self).__init__()
        self.args = args
        self.rgbenc = ImageEncoder(args)
        self.depthenc = ImageEncoder(args)
        depth_last_size = args.img_hidden_sz * args.num_image_embeds
        rgb_last_size = args.img_hidden_sz * args.num_image_embeds
        depth_rgb_last_size = args.img_hidden_sz * args.num_image_embeds * 2
        self.clf_depth = nn.ModuleList()
        self.clf_rgb = nn.ModuleList()
        self.clf_depth_rgb = nn.ModuleList()
        
        for hidden in args.hidden:
            self.clf_rgb.append(nn.Linear(rgb_last_size, hidden))
            print(rgb_last_size)
            self.clf_rgb.append(nn.ReLU())
            self.clf_rgb.append(nn.Dropout(args.dropout))
            rgb_last_size = hidden
        self.clf_rgb.append(nn.Linear(rgb_last_size, args.n_classes))
        
        for hidden in args.hidden:
            self.clf_depth.append(nn.Linear(depth_last_size, hidden))
            self.clf_depth.append(nn.ReLU())
            self.clf_depth.append(nn.Dropout(args.dropout))
            depth_last_size = hidden
        self.clf_depth.append(nn.Linear(depth_last_size, args.n_classes))
        
        for hidden in args.hidden:
            self.clf_depth_rgb.append(nn.Linear(depth_rgb_last_size, 2*hidden))
            self.clf_depth_rgb.append(nn.ReLU())
            self.clf_depth_rgb.append(nn.Dropout(args.dropout))
            depth_rgb_last_size = 2*hidden
        self.clf_depth_rgb.append(nn.Linear(depth_rgb_last_size, args.n_classes))


    def forward(self, rgb, depth):
        depth = self.depthenc(depth)
        depth = torch.flatten(depth, start_dim=1)
        rgb = self.rgbenc(rgb)
        rgb = torch.flatten(rgb, start_dim=1)
        
        rgb_out = rgb
        for layer in self.clf_rgb:
            rgb_out = layer(rgb_out)
        
        depth_out = depth
        for layer in self.clf_depth:
            depth_out = layer(depth_out)
        
        rgb_energy = -torch.logsumexp(rgb_out, dim=1)
        depth_energy = -torch.logsumexp(depth_out, dim=1)

        rgb_conf = -0.1*torch.reshape(rgb_energy, (-1,1))
        depth_conf = -0.1*torch.reshape(depth_energy, (-1,1))

        ### DYNAMIC LATE FUSION
        depth_rgb_out = (depth_out*depth_conf + rgb_out*rgb_conf)
        
        return  depth_rgb_out, rgb_out, depth_out, rgb_conf, depth_conf
        
    def get_feature(self, rgb, depth):
        depth = self.depthenc(depth)
        depth = torch.flatten(depth, start_dim=1)
        rgb = self.rgbenc(rgb)
        rgb = torch.flatten(rgb, start_dim=1)
        
        depth_out = depth
        for layer in self.clf_depth:
            depth_out = layer(depth_out)
        rgb_out = rgb
        for layer in self.clf_rgb:
            rgb_out = layer(rgb_out)
        
        return  rgb, depth, rgb_out, depth_out
