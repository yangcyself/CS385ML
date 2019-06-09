# -^- coding:utf-8 -^-
# this file writes the embeder class, which 
# has a model, forwarding the dataset to get the tensors
# and save the tensors in a tensorflow ckpt fashion
from collections import OrderedDict, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

class convEmbeder(object):
    """
    Please modify forward() and backward() according to your task.
    """

    def __init__(self, model,candidate_layers = None):
        super(convEmbeder, self).__init__()
        self.device = next(model.parameters()).device
        self.fmap_pool = OrderedDict()
        self.model = model
        self.handlers = []  # a set of hook function handlers
        self.candidate_layers = candidate_layers  # list

        def forward_hook(key):
            def forward_hook_(module, input, output):
                # Save featuremaps
                self.fmap_pool[key] = output.detach()

            return forward_hook_
        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(module.register_forward_hook(forward_hook(name)))


    def _compute_grad_weights(self, grads):
        return F.adaptive_avg_pool2d(grads, 1)


    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))


    def forward(self, image):
        """
        Simple classification
        """
        self.model.zero_grad()
        self.logits = self.model(image)
        self.probs = F.softmax(self.logits, dim=1)
        return self.probs.sort(dim=1, descending=True)


    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()

    def refresh(self):
        self.fmap_pool = OrderedDict()
        
    def generate(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer)
        weights = self._compute_grad_weights(fmaps)
        return weights
