DEFAULT_RUNS = {
    # all domains
    1: {'lexical_split': False, 'methods': ['elmo']},
    2: {'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    3: {'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    4: {'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    5: {'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    6: {'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},

    7: {'lexical_split': True, 'methods': ['elmo']},
    8: {'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    9: {'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    10: {'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    11: {'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    12: {'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},

    # in domain 112
    13: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo']},
    14: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    15: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    16: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    17: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    18: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},

    19: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo']},
    20: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    21: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    22: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    23: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    24: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},

    # in domain 114
    25: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo']},
    26: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    27: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    28: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    29: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    30: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},

    31: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo']},
    32: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    33: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    34: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    35: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    36: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},

    # in domain 115
    37: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo']},
    38: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    39: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    40: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    41: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    42: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},

    43: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo']},
    44: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    45: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    46: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    47: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    48: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},

    # out domain 112
    50: {'out_domain': '112', 'lexical_split': False, 'methods': ['elmo']},
    51: {'out_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    52: {'out_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    53: {'out_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    54: {'out_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    55: {'out_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},

    # out domain 114
    56: {'out_domain': '114', 'lexical_split': False, 'methods': ['elmo']},
    57: {'out_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    58: {'out_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    59: {'out_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    60: {'out_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    61: {'out_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},

    # out domain 115
    62: {'out_domain': '115', 'lexical_split': False, 'methods': ['elmo']},
    63: {'out_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    64: {'out_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    65: {'out_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    66: {'out_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    67: {'out_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
}

ALL_DOMAINS = {
    # all domains
    11: {'lexical_split': False, 'methods': ['elmo']},
    12: {'lexical_split': False, 'methods': ['elmo']},
    13: {'lexical_split': False, 'methods': ['elmo']},
    14: {'lexical_split': False, 'methods': ['elmo']},
    15: {'lexical_split': False, 'methods': ['elmo']},

    21: {'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    22: {'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    23: {'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    24: {'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    25: {'lexical_split': False, 'methods': ['elmo', 'fasttext']},

    31: {'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    32: {'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    33: {'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    34: {'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    35: {'lexical_split': False, 'methods': ['elmo', 'sent2vec']},

    41: {'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    42: {'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    43: {'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    44: {'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    45: {'lexical_split': False, 'methods': ['elmo', 'retrofit']},

    51: {'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    52: {'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    53: {'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    54: {'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    55: {'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},

    61: {'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    62: {'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    63: {'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    64: {'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    65: {'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
}

ALL_DOMAINS_LEXICAL = {
    71: {'lexical_split': True, 'methods': ['elmo']},
    72: {'lexical_split': True, 'methods': ['elmo']},
    73: {'lexical_split': True, 'methods': ['elmo']},
    74: {'lexical_split': True, 'methods': ['elmo']},
    75: {'lexical_split': True, 'methods': ['elmo']},

    81: {'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    82: {'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    83: {'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    84: {'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    85: {'lexical_split': True, 'methods': ['elmo', 'fasttext']},

    91: {'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    92: {'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    93: {'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    94: {'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    95: {'lexical_split': True, 'methods': ['elmo', 'sent2vec']},

    101: {'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    102: {'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    103: {'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    104: {'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    105: {'lexical_split': True, 'methods': ['elmo', 'retrofit']},

    111: {'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    112: {'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    113: {'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    114: {'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},

    121: {'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    122: {'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    123: {'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    124: {'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    125: {'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
}

IN_DOMAIN_LEXICAL_112 = {
    131: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo']},
    132: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo']},
    133: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo']},
    134: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo']},
    135: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo']},

    141: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    142: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    143: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    144: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    145: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},

    151: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    152: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    153: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    154: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    155: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},

    161: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    162: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    163: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    164: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    165: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},

    171: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    172: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    173: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    174: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    175: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},

    181: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    182: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    183: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    184: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    185: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
}

IN_DOMAIN_LEXICAL_114 = {
    191: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo']},
    192: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo']},
    193: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo']},
    194: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo']},
    195: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo']},

    201: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    202: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    203: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    204: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    205: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},

    211: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    212: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    213: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    214: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    215: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},

    221: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    222: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    223: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    224: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    225: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},

    231: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    232: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    233: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    234: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    235: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},

    241: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    242: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    243: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    244: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    245: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
}

IN_DOMAIN_LEXICAL_115 = {
    251: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo']},
    252: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo']},
    253: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo']},
    254: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo']},
    255: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo']},

    261: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    262: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    263: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    264: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    265: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},

    271: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    272: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    273: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    274: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    275: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},

    281: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    282: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    283: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    284: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    285: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},

    291: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    292: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    293: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    294: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    295: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},

    301: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    302: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    303: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    304: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    305: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
}

IN_DOMAIN_112 = {
    311: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo']},
    312: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo']},
    313: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo']},
    314: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo']},
    315: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo']},

    321: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    322: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    323: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    324: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    325: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},

    331: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    332: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    333: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    334: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    335: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},

    341: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    342: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    343: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    344: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    345: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},

    351: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    352: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    353: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    354: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    355: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},

    361: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    362: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    363: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    364: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    365: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
}

IN_DOMAIN_114 = {
    371: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo']},
    372: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo']},
    373: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo']},
    374: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo']},
    375: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo']},

    381: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    382: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    383: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    384: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    385: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},

    391: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    392: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    393: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    394: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    395: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},

    401: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    402: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    403: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    404: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    405: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},

    411: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    412: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    413: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    414: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    415: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},

    421: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    422: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    423: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    424: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    425: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
}

IN_DOMAIN_115 = {
    431: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo']},
    432: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo']},
    433: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo']},
    434: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo']},
    435: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo']},

    441: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    442: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    443: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    444: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    445: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},

    451: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    452: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    453: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    454: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    455: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},

    461: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    462: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    463: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    464: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    465: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},

    471: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    472: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    473: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    474: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    475: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},

    481: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    482: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    483: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    484: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    485: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
}

OUT_DOMAIN_112 = {
    # out domain 112
    50: {'out_domain': '112', 'lexical_split': False, 'methods': ['elmo']},
    51: {'out_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    52: {'out_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    53: {'out_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    54: {'out_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    55: {'out_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
}

OUT_DOMAIN_114 = {
    # out domain 114
    56: {'out_domain': '114', 'lexical_split': False, 'methods': ['elmo']},
    57: {'out_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    58: {'out_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    59: {'out_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    60: {'out_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    61: {'out_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
}

OUT_DOMAIN_115 = {
    # out domain 115
    62: {'out_domain': '115', 'lexical_split': False, 'methods': ['elmo']},
    63: {'out_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    64: {'out_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    65: {'out_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    66: {'out_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    67: {'out_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
}

RUNS = {
    'all': ALL_DOMAINS,
    'all_lexical': ALL_DOMAINS_LEXICAL,
    '112_in_lexical': IN_DOMAIN_LEXICAL_112,
    '114_in_lexical': IN_DOMAIN_LEXICAL_114,
    '115_in_lexical': IN_DOMAIN_LEXICAL_115,
    '112_in': IN_DOMAIN_112,
    '114_in': IN_DOMAIN_114,
    '115_in': IN_DOMAIN_115,
    '112_out': OUT_DOMAIN_112,
    '114_out': OUT_DOMAIN_114,
    '115_out': OUT_DOMAIN_115,
}
