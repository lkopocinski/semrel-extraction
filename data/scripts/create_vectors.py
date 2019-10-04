import argparse
import sys

import numpy as np
from allennlp.modules.elmo import Elmo, batch_to_ids
from parse_utils import Relation

np.set_printoptions(threshold=sys.maxsize)

try:
    import argcomplete
except ImportError:
    argcomplete = None


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
    with open(path, 'r', encoding='utf-8') as in_file:
        for line in in_file:
            relation = Relation(line)
            vector_from = get_context_vector(elmo, relation.source)
            vector_to = get_context_vector(elmo, relation.dest)

            print_line(relation_type, vector_from, vector_to, relation.source, relation.dest)


def get_context_vector(model, element):
    character_ids = batch_to_ids([element.context])
    embeddings = model(character_ids)
    v = embeddings['elmo_representations'][1].data.numpy()
    try:
        value = v[:, element.index, :].flatten()
    except:
        print(element, file=sys.stderr)
    return Vector(value)


def print_line(relation_type, vector_from, vector_to, relation_from, relation_to):
    print(f'{relation_type}\t{vector_from}\t{vector_to}\t{relation_from}\t{relation_to}')


def main(argv=None):
    args = get_args(argv)
    elmo = Elmo(args.options, args.weights, 2, dropout=0)
    create_vectors(elmo, args.source_file, args.relation_type)


if __name__ == "__main__":
    main()
