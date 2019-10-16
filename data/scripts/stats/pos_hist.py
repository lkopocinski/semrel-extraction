import os
from collections import defaultdict

import pandas as pd
from corpus_ccl import cclutils, token_utils, corpus_object_utils

hist_dict = defaultdict(lambda: defaultdict(int))

path = '../../korpusy/inforex_export_81/documents/'

for file_ in os.listdir(path):
    if file_.endswith('.xml') and not file_.endswith('.rel.xml'):
        doc = cclutils.read_ccl(path+file_)
        tokens = [(sent, token) for par in doc.paragraphs() for sent in par.sentences() for token in sent.tokens()]
        for ind, (sent, token) in enumerate(tokens):
            try:
                if token_utils.get_annotation(sent, token, 'BRAND_NAME') > 0:
                    continue
            except KeyError:
                continue
            left = []
            right = []
            try:
                left = tokens[ind - 3:ind]
            except Exception:
                pass
            try:
                right = tokens[ind + 1:ind + 4]
            except Exception:
                pass
            try:
                left = [corpus_object_utils.get_pos(t, 'nkjp') for _, t in left]
                pos = corpus_object_utils.get_pos(token, 'nkjp')
                right = [corpus_object_utils.get_pos(t, 'nkjp') for _, t in right]
            except TypeError:
                continue

            if pos in ['subst,nom,m3,sg', 'subst,gen,m3,sg', 'subst,loc,m3,sg', 'subst,nom,f,sg', 'brev,pun']:
                for pos_l in left:
                    hist_dict[pos_l]['left'] += 1

                hist_dict[pos]['center'] += 1

                for pos_r in right:
                    hist_dict[pos_r]['right'] += 1

df = pd.DataFrame.from_dict(hist_dict, orient='index', columns=['center', 'left', 'right'])
print(df)
