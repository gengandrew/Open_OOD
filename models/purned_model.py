# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import torch
import typing
import numpy as np

class PrunedModel(abc.ABC, torch.nn.Module):
    def __init__(self, model, mask):
        if isinstance(model, PrunedModel): raise ValueError('Cannot nest pruned models.')
        super(PrunedModel, self).__init__()

        self.model = model
        prunable_layer_names = [name + '.weight' for name, module in self.named_modules() if
                                isinstance(module, torch.nn.modules.conv.Conv2d) or
                                isinstance(module, torch.nn.modules.linear.Linear)]

        for k, v in mask.items(): self.register_buffer(PrunedModel.to_mask_name(k), v.float())
        self._apply_mask()

    @staticmethod
    def to_mask_name(name):
        return 'mask_' + name.replace('.', '___')
    
    def _apply_mask(self):
        for name, param in self.model.named_parameters():
            if hasattr(self, PrunedModel.to_mask_name(name)):
                param.data *= getattr(self, PrunedModel.to_mask_name(name)).cuda()

    def forward(self, x):
        self._apply_mask()
        return self.model.forward(x)

    def feature_list(self, x):
        return self.model.feature_list(x)

    def intermediate_forward(self, x, layer_index):
        return self.model.intermediate_forward(x, layer_index)

    def penultimate_forward(self, x):
        return self.model.penultimate_forward(x)
