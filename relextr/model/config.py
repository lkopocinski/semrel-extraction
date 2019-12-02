RUNS = {
    # all domains
    1: {'lexical_split': False, 'methods': ['elmo']},
    2: {'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    3: {'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    4: {'lexical_split': True, 'methods': ['elmo']},
    5: {'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    6: {'lexical_split': True, 'methods': ['elmo', 'retrofit']},

    # in domain 112
    7: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo']},
    8: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    9: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    10: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo']},
    11: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    12: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},

    # in domain 114
    13: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo']},
    14: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    15: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    16: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo']},
    17: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    18: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},

    # in domain 115
    19: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo']},
    20: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    21: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    22: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo']},
    23: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    24: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},

    # out domain 112
    25: {'out_domain': '112', 'lexical_split': False, 'methods': ['elmo']},
    26: {'out_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    27: {'out_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},

    # out domain 114
    28: {'out_domain': '114', 'lexical_split': False, 'methods': ['elmo']},
    29: {'out_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    30: {'out_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},

    # out domain 115
    31: {'out_domain': '115', 'lexical_split': False, 'methods': ['elmo']},
    32: {'out_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    33: {'out_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
}
