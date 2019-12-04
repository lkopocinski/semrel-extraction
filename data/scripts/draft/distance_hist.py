import pandas as pd
from pathlib import Path

from data.scripts.utils.corpus import id_to_sent_dict, is_ner_relation, is_in_channel, get_relation_element, \
    documents_gen


def distance_hist(relation_files, channels, nr):
    to_save = []

    files = []
    with open(relation_files, 'r', encoding='utf-8') as f:
        for line in f:
            files.append(Path(line.strip()))

    for document in documents_gen(files):
        sentences = id_to_sent_dict(document)

        for relation in document.relations():
            if is_ner_relation(relation):
                if is_in_channel(relation, channels):
                    f = relation.rel_from()
                    t = relation.rel_to()

                    f_element = get_relation_element(f, sentences)
                    t_element = get_relation_element(t, sentences)

                    if f_element and t_element:
                        if f_element.context == t_element.context:
                            if t_element.indices[0] > f_element.indices[0]:
                                shift = (len(f_element.indices) - 1) if len(f_element.indices) > 1 else 0
                                dist = t_element.indices[0] - (f_element.indices[0] + shift)
                                dist = abs(dist)
                            elif f_element.indices[0] > t_element.indices[0]:
                                shift = (len(t_element.indices) - 1) if len(t_element.indices) > 1 else 0
                                dist = f_element.indices[0] - (t_element.indices[0] + shift)
                                dist = abs(dist)
                            else:
                                dist = 0
                            to_save.append(dist)

    hist = pd.Series(to_save)
    hist = hist.value_counts()
    hist = hist.sort_index()
    print(hist.sum())
    hist.to_csv(f'{nr}.dist.hist', sep='\t', header=False)


if __name__ == '__main__':
    for nr in [112, 114, 115]:
        distance_hist(f'{nr}.files', ["BRAND_NAME", "PRODUCT_NAME"], nr)
