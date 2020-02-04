from collections import defaultdict
from pathlib import Path
from typing import List
import pandas as pd

import click
from corpus_ccl import token_utils as tou

from data.scripts.utils.corpus import relations_documents_gen, id_to_sent_dict, is_ner_relation, is_in_channel, \
    get_relation_element, \
    get_lemma
from data.scripts.utils.io import read_lines


def generate(relation_files: List[Path], channels):
    brands_solo = defaultdict(int)

    for document in relations_documents_gen(relation_files):
        sentences = id_to_sent_dict(document)
        in_relation = set()

        for relation in document.relations():
            if is_ner_relation(relation) and is_in_channel(relation, channels):
                f = relation.rel_from()
                t = relation.rel_to()
                f_sent_id = f.sentence_id()
                t_sent_id = t.sentence_id()

                f_element = get_relation_element(f, sentences)
                t_element = get_relation_element(t, sentences)
                f_indices = tuple(f_element.indices)
                t_indices = tuple(t_element.indices)

                for i in f_indices:
                    in_relation.add((f_sent_id, i))
                for j in t_indices:
                    in_relation.add((t_sent_id, j))

        for par in document.paragraphs():
            for sentence in par.sentences():
                for idx, token in enumerate(sentence.tokens()):
                    if tou.get_annotation(sentence, token, "BRAND_NAME", default=0) != 0 \
                            and (sentence.id(), idx) not in in_relation:
                        lemma = get_lemma(sentence, idx)
                        brands_solo[lemma] += 1

    return brands_solo


@click.command()
@click.option(
    '--fileindex',
    required=True,
    help='File with corpus files.'
)
def main(fileindex):
    paths = [Path(line) for line in list(read_lines(Path(fileindex)))]
    brands_solo = generate(paths, ('BRAND_NAME', 'PRODUCT_NAME'))

    df = pd.Series(brands_solo).to_frame('ALONE').sort_values(by=['ALONE'], ascending=False)
    df.to_csv(f'{fileindex}.freq', sep='\t')


if __name__ == '__main__':
    main()
