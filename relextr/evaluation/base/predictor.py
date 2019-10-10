import numpy as np


class Predictor(object):

    def __init__(self, net_model, emb_model):
        self._emb_model = emb_model
        self._net_model = net_model

    def predict(self, data):
        (f_idx, f_ctx), (s_idx, s_ctx) = data

        f_v = self._emb_model.embedd(f_ctx, f_idx)
        s_v = self._emb_model.embedd(s_ctx, s_idx)

        v = np.concatenate([f_v, s_v])

        return self._net_model.predict(v)
