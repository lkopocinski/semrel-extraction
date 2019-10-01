from allennlp.modules.elmo import Elmo, batch_to_ids
import re
import argparse


import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

try:
    import argcomplete
except ImportError:
    argcomplete = None

"""
Change options_file and weight_file to path with elmo data

"""
options_file = '/data2/piotrmilkowski/bilm-tf-data/e2000000/options.json'
weights_file = '/data2/piotrmilkowski/bilm-tf-data/e2000000/weights.hdf5'

elmo = Elmo(options_file, weights_file, 2, dropout=0)

def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source_file', required=True, help="A file with relations contexts.")
    parser.add_argument('-r', '--relation_type', required=True, help="Example's relation type.")
    if argcomplete:
        argcomplete.autocomplete(parser)
    
    return parser.parse_args(argv)


def print_example(relation_type, vector_from, vector_to, phrase_from, phrase_to):
    print('{}\t{}\t{}\t{}\t{}'.format(relation_type, vector_from, vector_to, phrase_from, phrase_to))


def print_example_with_ctx(relation_type, vector_from, vector_to, idx_from, idx_to, context_from, context_to):
    print(f'{relation_type}\t{vector_from}\t{vector_to}\t{context_from[idx_from]}\t{context_to[idx_to]}\t{idx_from}:{context_from}\t{idx_to}:{context_to}')


def contextual_vector(context, model, idx):
    """
    Creating elmo vector for context (sentence). 
    idx is a id of meaning word (Brand or Product) in context.
    Parametres
    ----------
    context : str
        sentence with brand or product marked
    model : elmo model
        previous created model Elmo
    idx : int
        index of word for creating contextual vector
    """
    character_ids = batch_to_ids([context])
    embeddings = model(character_ids)
    v = embeddings['elmo_representations'][1].data.numpy()
    return v[:,idx,:].flatten()


def create_vectors(path, relation):
    with open(path) as f:
        for line in f:
            line = line.strip()
            by_tab = line.split('\t')
            
            idx_from, ctx_from = by_tab[0].split(':', 1)
            idx_to, ctx_to = by_tab[1].split(':', 1)
            
            ctx_from = eval(ctx_from)
            ctx_to = eval(ctx_to)

            idx_from = int(idx_from)
            idx_to = int(idx_to)

            vector_from = contextual_vector(ctx_from, elmo, idx_from)
            vector_to = contextual_vector(ctx_to, elmo, idx_to)
            
            vector_from = np.array2string(vector_from, separator=', ').replace('\n', '')
            vector_to =  np.array2string(vector_to, separator=', ').replace('\n', '')

            print_example_with_ctx(relation, vector_from, vector_to, idx_from, idx_to, ctx_from, ctx_to)


def main(argv=None):
    args = get_args(argv)
    create_vectors(args.source_file, args.relation_type)


if __name__ == "__main__":
    main()
