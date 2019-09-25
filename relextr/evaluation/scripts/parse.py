#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from corpus_ccl import cclutils

from relextr.evaluation.base import Parser
from relextr.evaluation.base import Predictor
from relextr.evaluation.scripts.embutils import ElmoEmb
from relextr.model import RelNet

import argparse

try:
    import argcomplete
except ImportError:
    argcomplete = None


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--net_model', required=True,
                        help="A neural model for BRAND - PRODUCT recognition")
    parser.add_argument('-e', '--emb_model', required=True,
                        help="A path to embedding model, compatible with "
                        "neural model (`--net_model` parametere.")
    parser.add_argument('-b', '--batch', required=True,
                        help="A path to the list of CCL files to process")
    if argcomplete:
        argcomplete.autocomplete(parser)
    return parser.parse_args(argv)


def load_data(datalist):
    with open(datalist) as ifile:
        for line in ifile:
            ccl_path = line.strip()
            if not os.path.exists(ccl_path):
                continue
            yield cclutils.read_ccl(ccl_path)


def main(argv=None):
    args = get_args(argv)
    net_model = RelNet()
    net_model.load(args.net_model)

    emb_model = ElmoEmb(args.emb_model)

    predictor = Predictor(net_model, emb_model)
    parser = Parser()

    for doc in load_data(args.batch):
        for sample in parser(doc):
            decision = predictor.predict(sample)
            (f_idx, f_ctx), (s_idx, s_ctx) = sample
            print('{}\t{}: {}'.format(f_ctx[f_idx].encode('utf-8'), s_ctx[s_idx].encode('utf-8'), decision))


if __name__ == "__main__":
    main()
