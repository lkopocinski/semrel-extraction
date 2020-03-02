DEFAULT_RUNS = {
    # all domains
    1001: {'lexical_split': False, 'methods': ['elmo']},
    1002: {'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    1003: {'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    1004: {'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    1005: {'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    1006: {'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},

    1007: {'lexical_split': True, 'methods': ['elmo']},
    1008: {'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    1009: {'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    1000: {'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    1011: {'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    1012: {'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},

    # in domain 112
    1013: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo']},
    1014: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    1015: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    1016: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    1017: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    1018: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},

    1019: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo']},
    1020: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    1021: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    1022: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    1023: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    1024: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},

    # in domain 114
    1025: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo']},
    1026: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    1027: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    1028: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    1029: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    1030: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},

    1031: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo']},
    1032: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    1033: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    1034: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    1035: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    1036: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},

    # in domain 115
    1037: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo']},
    1038: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    1039: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    1040: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    1041: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    1042: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},

    1043: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo']},
    1044: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    1045: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    1046: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    1047: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    1048: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},

    # # out domain 112
    # 1050: {'out_domain': '112', 'lexical_split': False, 'methods': ['elmo']},
    # 1051: {'out_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    # 1052: {'out_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    # 1053: {'out_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    # 1054: {'out_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    # 1055: {'out_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    #
    # # out domain 114
    # 1056: {'out_domain': '114', 'lexical_split': False, 'methods': ['elmo']},
    # 1057: {'out_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    # 1058: {'out_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    # 1059: {'out_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    # 1060: {'out_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    # 1061: {'out_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    #
    # # out domain 115
    # 1062: {'out_domain': '115', 'lexical_split': False, 'methods': ['elmo']},
    # 1063: {'out_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    # 1064: {'out_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    # 1065: {'out_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    # 1066: {'out_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    # 1067: {'out_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
}

ALL_DOMAINS = {
    # all domains
    2001: {'lexical_split': False, 'methods': ['elmo']},
    2002: {'lexical_split': False, 'methods': ['elmo']},
    2003: {'lexical_split': False, 'methods': ['elmo']},
    2004: {'lexical_split': False, 'methods': ['elmo']},
    2005: {'lexical_split': False, 'methods': ['elmo']},

    2006: {'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    2007: {'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    2008: {'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    2009: {'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    2010: {'lexical_split': False, 'methods': ['elmo', 'fasttext']},

    2011: {'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    2012: {'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    2013: {'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    2014: {'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    2015: {'lexical_split': False, 'methods': ['elmo', 'sent2vec']},

    2016: {'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    2017: {'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    2018: {'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    2019: {'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    2020: {'lexical_split': False, 'methods': ['elmo', 'retrofit']},

    2021: {'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    2022: {'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    2023: {'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    2024: {'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    2025: {'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},

    2026: {'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    2027: {'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    2028: {'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    2029: {'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    2030: {'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
}

ALL_DOMAINS_LEXICAL = {
    3001: {'lexical_split': True, 'methods': ['elmo']},
    3002: {'lexical_split': True, 'methods': ['elmo']},
    3003: {'lexical_split': True, 'methods': ['elmo']},
    3004: {'lexical_split': True, 'methods': ['elmo']},
    3005: {'lexical_split': True, 'methods': ['elmo']},

    3006: {'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    3007: {'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    3008: {'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    3009: {'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    3010: {'lexical_split': True, 'methods': ['elmo', 'fasttext']},

    3011: {'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    3012: {'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    3013: {'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    3014: {'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    3015: {'lexical_split': True, 'methods': ['elmo', 'sent2vec']},

    3016: {'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    3017: {'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    3018: {'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    3019: {'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    3020: {'lexical_split': True, 'methods': ['elmo', 'retrofit']},

    3021: {'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    3022: {'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    3023: {'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    3024: {'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    3025: {'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},

    3026: {'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    3027: {'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    3028: {'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    3029: {'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    3030: {'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
}

IN_DOMAIN_LEXICAL_112 = {
    4001: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo']},
    4002: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo']},
    4003: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo']},
    4004: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo']},
    4005: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo']},

    4006: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    4007: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    4008: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    4009: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    4010: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},

    4011: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    4012: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    4013: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    4014: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    4015: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},

    4016: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    4017: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    4018: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    4019: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    4020: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},

    4021: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    4022: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    4023: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    4024: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    4025: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},

    4026: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    4027: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    4028: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    4029: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    4030: {'in_domain': '112', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
}

IN_DOMAIN_LEXICAL_114 = {
    5001: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo']},
    5002: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo']},
    5003: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo']},
    5004: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo']},
    5005: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo']},

    5006: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    5007: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    5008: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    5009: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    5010: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},

    5011: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    5012: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    5013: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    5014: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    5015: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},

    5016: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    5017: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    5018: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    5019: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    5020: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},

    5021: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    5022: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    5023: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    5024: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    5025: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},

    5026: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    5027: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    5028: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    5029: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    5030: {'in_domain': '114', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
}

IN_DOMAIN_LEXICAL_115 = {
    6001: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo']},
    6002: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo']},
    6003: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo']},
    6004: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo']},
    6005: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo']},

    6006: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    6007: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    6008: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    6009: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},
    6010: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'fasttext']},

    6011: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    6012: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    6013: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    6014: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},
    6015: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'sent2vec']},

    6016: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    6017: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    6018: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    6019: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},
    6020: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'retrofit']},

    6021: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    6022: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    6023: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    6024: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    6025: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'fasttext', 'sent2vec']},

    6026: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    6027: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    6028: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    6029: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    6030: {'in_domain': '115', 'lexical_split': True, 'methods': ['elmo', 'retrofit', 'sent2vec']},
}

IN_DOMAIN_112 = {
    7001: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo']},
    7002: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo']},
    7003: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo']},
    7004: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo']},
    7005: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo']},

    7006: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    7007: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    7008: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    7009: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    7010: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},

    7011: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    7012: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    7013: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    7014: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    7015: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},

    7016: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    7017: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    7018: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    7019: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    7020: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},

    7021: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    7022: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    7023: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    7024: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    7025: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},

    7026: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    7027: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    7028: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    7029: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    7030: {'in_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
}

IN_DOMAIN_114 = {
    8001: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo']},
    8002: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo']},
    8003: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo']},
    8004: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo']},
    8005: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo']},

    8006: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    8007: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    8008: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    8009: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    8010: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},

    8011: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    8012: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    8013: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    8014: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    8015: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},

    8016: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    8017: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    8018: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    8019: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    8020: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},

    8021: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    8022: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    8023: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    8024: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    8025: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},

    8026: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    8027: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    8028: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    8029: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    8030: {'in_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
}

IN_DOMAIN_115 = {
    9001: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo']},
    9002: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo']},
    9003: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo']},
    9004: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo']},
    9005: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo']},

    9006: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    9007: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    9008: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    9009: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    9010: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},

    9011: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    9012: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    9013: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    9014: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    9015: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},

    9016: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    9017: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    9018: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    9019: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    9020: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},

    9021: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    9022: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    9023: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    9024: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    9025: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},

    9026: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    9027: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    9028: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    9029: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
    9030: {'in_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
}

OUT_DOMAIN_112 = {
    # out domain 112
    10001: {'out_domain': '112', 'lexical_split': False, 'methods': ['elmo']},
    10002: {'out_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    10003: {'out_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    10004: {'out_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    10005: {'out_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    10006: {'out_domain': '112', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
}

OUT_DOMAIN_114 = {
    # out domain 114
    20001: {'out_domain': '114', 'lexical_split': False, 'methods': ['elmo']},
    20002: {'out_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    20003: {'out_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    20004: {'out_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    20005: {'out_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    20006: {'out_domain': '114', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
}

OUT_DOMAIN_115 = {
    # out domain 115
    30001: {'out_domain': '115', 'lexical_split': False, 'methods': ['elmo']},
    30002: {'out_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'fasttext']},
    30003: {'out_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'sent2vec']},
    30004: {'out_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'retrofit']},
    30005: {'out_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'fasttext', 'sent2vec']},
    30006: {'out_domain': '115', 'lexical_split': False, 'methods': ['elmo', 'retrofit', 'sent2vec']},
}

RUNS = {
    'default': DEFAULT_RUNS,
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
