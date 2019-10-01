import argparse
import sys

import numpy as np
from allennlp.modules.elmo import Elmo, batch_to_ids

np.set_printoptions(threshold=sys.maxsize)

try:
    import argcomplete
except ImportError:
    argcomplete = None


class Relation:

    def __init__(self, line):
        self.line = line
        self._element_from = None
        self._element_to = None
        self.extract_relation()

    def extract_relation(self):
        line = self.line.strip()
        by_tab = line.split('\t')

        lemma_from, lemma_to = by_tab[0].replace(' : ', ':').split(':', 1)
        channel_from, channel_to = by_tab[1].replace(' : ', ':').split(':', 1)
        idx_from, ctx_from = by_tab[2].split(':', 1)
        idx_to, ctx_to = by_tab[3].split(':', 1)

        ctx_from = eval(ctx_from)
        ctx_to = eval(ctx_to)

        idx_from = int(idx_from)
        idx_to = int(idx_to)

        self._element_from = self.Element(lemma_from, channel_from, ctx_from, idx_from)
        self._element_to = self.Element(lemma_to, channel_to, ctx_to, idx_to)

    @property
    def source(self):
        return self._element_from

    @property
    def dest(self):
        return self._element_to

    class Element:
        def __init__(self, lemma, channel, index, context):
            self.lemma = lemma
            self.channel = channel
            self.index = index
            self.context = context

        def __str__(self):
            return f'{self.lemma}\t{self.channel}\t{self.index}:{self.context}'


class Vector:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return np.array2string(self.value, separator=', ').replace('\n', '')


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source_file', required=True, help="A file with relations contexts.")
    parser.add_argument('-r', '--relation_type', required=True, help="Example's relation type.")
    parser.add_argument('-w', '--weights', required=True, help="File with weights to elmo model.")
    parser.add_argument('-p', '--options', required=True, help="File with options to elmo model.")

    if argcomplete:
        argcomplete.autocomplete(parser)

    return parser.parse_args(argv)


def create_vectors(elmo, path, relation_type):
    with open(path) as f:
        for line in f:
            relation = Relation(line)
            vector_from = contextual_vector(elmo, relation.source)
            vector_to = contextual_vector(elmo, relation.dest)

            print_line(relation_type, vector_from, vector_to, relation.source, relation.dest)


def contextual_vector(model, element):
    character_ids = batch_to_ids([element.context])
    embeddings = model(character_ids)
    v = embeddings['elmo_representations'][1].data.numpy()
    value = v[:, element.index, :].flatten()
    return Vector(value)


def print_line(relation_type, vector_from, vector_to, relation_from, relation_to):
    print(f'{relation_type}\t{vector_from}\t{vector_to}\t{relation_from}\t{relation_to}')


def main(argv=None):
    args = get_args(argv)
    elmo = Elmo(args.options, args.weights, 2, dropout=0)
    create_vectors(elmo, args.source_file, args.relation_type)


if __name__ == "__main__":
    main()
