import os

import click
import torch
from corpus_ccl import cclutils

from data.scripts.utils.vectorizers import ElmoVectorizer, FastTextVectorizer
from relextr.model.scripts.relnet import RelNet
from relextr.evaluation.scripts.extractor import Parser


class Predictor(object):

    def __init__(self, net_model, elmo, fasttext):
        pass
#        self._net_model = net_model
#        self._elmo = elmo
#        self._fasttext = fasttext

    def predict(self, data):
        (idx1, ctx1), (idx2, ctx2) = data

        ev1 = self._elmo.embed(ctx1)[idx1]
        ev2 = self._elmo.embed(ctx2)[idx2]

        fv1 = self._fasttext.embed(ctx1)[idx1]
        fv2 = self._fasttext.embed(ctx2)[idx2]

        v = torch.cat([ev1, ev2, fv1, fv2])

        return self._net_model.predict(v)


def documents(fileindex):
    with open(fileindex, 'r', encoding='utf-8') as f:
          paths = [line.strip() for line in f if os.path.exists(line.strip())]
    return (cclutils.read_ccl(path) for path in paths)


@click.command()
@click.option('--net_model', required=True, type=str, help="A neural model for BRAND - PRODUCT recognition")
@click.option('--elmo_model', required=True, type=(str, str), help="A path to elmo model options, weight")
@click.option('--fasttext_model', required=True, type=str, help="A path to fasttext model")
@click.option('--fileindex', required=True, type=str, help="A path to the list of CCL files to process")
def main(net_model, elmo_model, fasttext_model, fileindex):
 #   net = RelNet(in_dim=2648)
 #   net.load(net_model)

 #   elmo = ElmoVectorizer(*elmo_model)
 #   fasttext = FastTextVectorizer(fasttext_model)

 #   predictor = Predictor(net_model, elmo, fasttext)
    parser = Parser()

    for doc in documents(fileindex):
        ccl_path = doc.path().split(';')[0]
        out_path = f'{ccl_path}.txt'

        with open(out_path, 'w', encoding='utf-8') as f:
            for sample in parser(doc):
#                decision = predictor.predict(sample)
                (f_idx, f_ctx), (s_idx, s_ctx) = sample
                f.write(f'{f_ctx[f_idx]}\t{s_ctx[s_idx]}: {decision}\n')


if __name__ == '__main__':
    main()
