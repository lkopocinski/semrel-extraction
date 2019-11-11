#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os

from corpus_ccl import cclutils

from relextr.evaluation.base import Parser
from relextr.evaluation.base import Predictor

from relextr.model.scripts import RelNet

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


def lines(pathfilename):
    with open(pathfilename) as f:
        return [line.strip() for line in f if line.strip()]


def load_data(datalist):
    inputs = lines(datalist)
    outputs = [f'{path}.txt' for path in inputs]
    return zip(inputs, outputs)


def main(argv=None):
    args = get_args(argv)
    net_model = RelNet()
    net_model.load(args.net_model)

    emb_model = ElmoEmb(args.emb_model)

    predictor = Predictor(net_model, emb_model)
    parser = Parser()

    for in_path, out_path in load_data(args.batch):
        if os.path.exists(in_path):
            doc = cclutils.read_ccl(in_path)
            with open(out_path, 'w', encoding='utf-8') as out_file:
                for sample in parser(doc):
                    decision = predictor.predict(sample)
                    (f_idx, f_ctx), (s_idx, s_ctx) = sample
                    out_file.write(f'{f_ctx[f_idx]}\t{s_ctx[s_idx]}: {decision}\n')


if __name__ == "__main__":
    main()
