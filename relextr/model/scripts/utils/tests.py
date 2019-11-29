#!/usr/bin/env python
# -*- coding: utf-8 -*-
from batches import Dataset, Sampler


def main():
    keys = {
        1: ('111', 'in_relation', '17', 'sent1', 'sent1', 'BRAND_NAME', 'PRODUCT_NAME'),
        2: ('111', 'no_relation', '23', 'sent1', 'sent2', 'BRAND_NAME', 'PRODUCT_NAME'),
        3: ('111', 'in_relation', '48', 'sent1', 'sent1', 'BRAND_NAME', 'PRODUCT_NAME'),
        4: ('112', 'no_relation', '1', 'sent1', 'sent1', 'BRAND_NAME', 'PRODUCT_NAME'),
        5: ('112', 'in_relation', '2', 'sent1', 'sent2', 'BRAND_NAME', 'PRODUCT_NAME'),
        6: ('112', 'in_relation', '2', 'sent3', 'sent4', 'BRAND_NAME', 'PRODUCT_NAME'),
        7: ('113', 'in_relation', '11', 'sent1', 'sent1', 'BRAND_NAME', 'PRODUCT_NAME'),
        8: ('113', 'in_relation', '12', 'sent1', 'sent1', 'BRAND_NAME', 'PRODUCT_NAME'),
        9: ('113', 'no_relation', '13', 'sent3', 'sent3', 'BRAND_NAME', 'PRODUCT_NAME'),
    }

    models = None

    dataset = Dataset(models, keys)


if __name__ == "__main__":
    main()
