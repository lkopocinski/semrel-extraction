DEFAULT_RUNS = {
    # all domains
    # # 1: {'lexical_split': False, 'methods': ['elmo']},
    # # 2: {'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    #    3: {'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    # # 4: {'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    #    5: {'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    #    6: {'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},

    #    7: {'lexical_split': True, 'methods': ['elmo']},
    #    8: {'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    #    9: {'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    #    10: {'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    #    11: {'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    #    12: {'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},

    # in domain 112
    # 13: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo']},
    # 14: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    #    15: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    # 16: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    #    17: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    #    18: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},

    #    19: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo']},
    #    20: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    #    21: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    ###    22: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    #    23: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    ###    24: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},

    # in domain 114
    # 25: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo']},
    # 26: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    #    27: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    # 28: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    #    29: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    #    30: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},

    #    31: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo']},
    #    32: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    ###    33: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    ###    34: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    ###    35: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    ###    36: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},

    # in domain 115
    # 37: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo']},
    # 38: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    #    39: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    # 40: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    #    41: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    #    42: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},

    #    43: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo']},
    44: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    45: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    46: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    47: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    48: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},

    # out domain 112
    # 50: {'out_domain': '112', 'lexical_split': False, 'methods': ['elmo']},
    # 51: {'out_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    52: {'out_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    # 53: {'out_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    54: {'out_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    55: {'out_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},

    # out domain 114
    # 56: {'out_domain': '114', 'lexical_split': False, 'methods': ['elmo']},
    # 57: {'out_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    58: {'out_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    # 59: {'out_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    60: {'out_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    61: {'out_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},

    # out domain 115
    # 62: {'out_domain': '115', 'lexical_split': False, 'methods': ['elmo']},
    # 63: {'out_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    64: {'out_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    # 65: {'out_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    66: {'out_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    67: {'out_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
}

AREK_RUNS_112 = {
    1: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo']},
    2: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo']},
    3: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo']},
    4: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo']},
    5: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo']},

    6: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    7: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    8: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    9: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    10: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},

    11: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    12: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    13: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    14: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    15: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},

    16: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    17: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    18: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    19: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    20: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},

    21: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    22: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    23: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    24: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    25: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},

    26: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    27: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    28: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    29: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    30: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
}

AREK_RUNS_114 = {
    1: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo']},
    2: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo']},
    3: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo']},

    6: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    7: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    8: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},

    11: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    12: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    13: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},

    16: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    17: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    18: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},

    21: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    22: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    23: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},

    26: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    27: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    28: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
}

AREK_RUNS_115 = {
    1: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo']},
    2: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo']},
    3: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo']},

    6: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    7: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    8: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},

    11: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    12: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    13: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},

    16: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    17: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    18: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},

    21: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    22: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    23: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},

    26: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    27: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    28: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
}

RUNS = {
    '112': AREK_RUNS_112,
    '114': AREK_RUNS_114,
    '115': AREK_RUNS_115,
}
