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
    net_model.load('/mnt/lvm_work/REPOS/semrel-extraction/models/net-model.pt')

    emb_model = ElmoEmb(
        '/mnt/lvm_work/REPOS/semrel-extraction/models/emb-options.json',
        '/mnt/lvm_work/REPOS/semrel-extraction/models/emb-weights.hdf5',
    )

    predictor = Predictor(net_model, emb_model)
    parser = Parser()

    doc = cclutils.read_ccl('/mnt/lvm_work/REPOS/semrel-extraction/data/input.xml')
    for sample in parser(doc):
        print sample
        # decision = predictor.predict(sample)


if __name__ == "__main__":
    main()
