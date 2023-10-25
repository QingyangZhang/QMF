#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from src.models.bert import BertClf
from src.models.bow import GloveBowClf
from src.models.concat_bert import MultimodalConcatBertClf
from src.models.concat_bow import  MultimodalConcatBowClf
from src.models.image import ImageClf
from src.models.mmbt import MultimodalBertClf
from src.models.late_fusion import MultimodalLateFusionClf
from src.models.tmc import TMC,ce_loss
MODELS = {
    "bert": BertClf,
    "bow": GloveBowClf,
    "concatbow": MultimodalConcatBowClf,
    "concatbert": MultimodalConcatBertClf,
    "img": ImageClf,
    "mmbt": MultimodalBertClf,
    'latefusion':MultimodalLateFusionClf,
    'tmc':TMC
}


def get_model(args):
    # print(args.model)
    return MODELS[args.model](args)
