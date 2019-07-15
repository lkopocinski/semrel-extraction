#!/usr/bin/env python
# -*- coding: utf-8 -*-
# import sys
# import argparse

from corpus_ccl import cclutils

from relextr.base import Parser
from relextr.base import Predictor
from relextr.utils.embutils import ElmoEmb
from relextr.model import RelNet


def main():
    net_model = RelNet()
    net_model.load('../../models/net-model.pt')

    emb_model = ElmoEmb(
        '../../models/emb-options.json',
        '../../models/emb-weights.hdf5',
    )

    predictor = Predictor(net_model, emb_model)
    parser = Parser()

    doc = cclutils.read_ccl('../../data/input.xml')
    for sample in parser(doc):
        print(sample, file=open('./test.txt', mode='w', encoding='utf-8'))
        decision = predictor.predict(sample)
        (f_idx, f_ctx), (s_idx, s_ctx) = sample
        print('{}\t{}: {}'.format(f_ctx[f_idx], s_ctx[s_idx], decision), file=open('./out.txt', mode='w', encoding='utf-8'))


if __name__ == "__main__":
    main()
