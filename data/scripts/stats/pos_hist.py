import os

from corpus_ccl import cclutils, token_utils, corpus_object_utils

for file_ in os.listdir('./'):
    if file_.endswith('.xml') and not file_.endswith('.rel.xml'):
        doc = cclutils.read_ccl(file_)
        tokens = [(sent, token) for par in doc.paragraphs() for sent in par.sentences() for token in sent.tokens()]
        for ind, (sent, token) in enumerate(tokens):
            try:
                if token_utils.get_annotation(sent, token, 'BRAND_NAME') <= 0:
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
                left = [(corpus_object_utils.get_pos(t, 'nkjp'), t.orth_utf8()) for _, t in left]
                pos = (corpus_object_utils.get_pos(token, 'nkjp'), token.orth_utf8())
                right = [(corpus_object_utils.get_pos(t, 'nkjp'), t.orth_utf8()) for _, t in right]
            except TypeError:
                continue
            print(left, pos, right)
