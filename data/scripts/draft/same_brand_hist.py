import pandas as pd
from pathlib import Path
from collections import defaultdict

from data.scripts.utils.corpus import id_to_sent_dict, is_ner_relation, is_in_channel, get_relation_element, \
    documents_gen


def same_brand_hist(relation_files, channels, nr):
    sizes = []

    files = []
    with open(relation_files, 'r', encoding='utf-8') as f:
        for line in f:
            files.append(Path(line.strip()))

    for document in documents_gen(files):
        brand_dict = defaultdict(int)

        sentences = id_to_sent_dict(document)
        for relation in document.relations():
            if is_ner_relation(relation):
                if is_in_channel(relation, channels):
                    f = relation.rel_from()
                    t = relation.rel_to()

                    f_element = get_relation_element(f, sentences)
                    t_element = get_relation_element(t, sentences)

                    if f_element and t_element:
                        if f_element.channel == "BRAND_NAME":
                            brand_dict[f_element.lemma] += 1
                        elif t_element.channel == "BRAND_NAME":
                            brand_dict[t_element.lemma] += 1

        brands = set(brand_dict.keys())
        size = len(brands)
        sizes.append(size)

    hist = pd.Series(sizes)
    hist = hist.value_counts()
    hist = hist.sort_index()
    hist.to_csv(f'{nr}.multi.brands.hist', sep='\t', header=False)


if __name__ == '__main__':
    for nr in [112, 114, 115]:
        same_brand_hist(f'{nr}.files', ["BRAND_NAME", "PRODUCT_NAME"], nr)
