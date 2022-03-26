#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as utils_rnn
import os
import time



HIDDEN_SIZE = 128
NUM_CLASSES = 6


class ImageRNN(nn.Module):

    def __init__(self, input_size, hidden_size, nclasses, bidirectional=False):
        super(ImageRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nclasses = nclasses
        self.bid = bidirectional

        self.gru = nn.GRU(input_size, hidden_size, bidirectional=self.bid)
        self.fc = nn.Linear(hidden_size, nclasses)

    def forward(self, s_nimgs, hidden=None):
        s_timesteps = np.ones(s_nimgs.shape[0], dtype=np.int64) * 10

        s_nimgs = torch.transpose(s_nimgs, 0, 1)
        s_nimgs_p = utils_rnn.pack_padded_sequence(s_nimgs, s_timesteps)

        output_pack, hidden = self.gru(s_nimgs_p, hidden)
        output, output_len = utils_rnn.pad_packed_sequence(output_pack)

        # print("out", output.shape)
        # print("hidden", hidden.shape)
        if self.bid:
            last_hidden = hidden[0, :, :] + hidden[1, :, :]
        else:
            last_hidden = hidden.squeeze()
        out = self.fc(last_hidden)
        return torch.sigmoid(out), hidden
