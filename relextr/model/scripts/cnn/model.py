import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self,
                 emb_dim,
                 n_filters,
                 filter_sizes,
                 out_dim,
                 dropout):
        super(CNN, self).__init__()

        self.convolutions = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fsize, emb_dim))
            for fsize in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embedded):
        # embedded: batch, sent, emb
        embedded = embedded.unsqueeze(1)
        # embedded: batch, 1, sent, emb
        convoluted = [
            F.relu(conv(embedded)).squeeze(3)
            for conv in self.convolutions
        ]
        pooled = [
            F.max_pool1d(conv, conv.shape[2]).squeeze(2)
            for conv in convoluted
        ]

        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

