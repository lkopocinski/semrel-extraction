#!/usr/bin/env python
# -*- coding: utf-8 -*-
from batches import Dataset, Sampler
from torch.utils.data import DataLoader


def main():
    keys = {
        1: ('111', 'in_relation', '17', 'sent1', 'sent1', 'BRAND_NAME', 'PRODUCT_NAME'),
        2: ('111', 'no_relation', '23', 'sent1', 'sent2', 'BRAND_NAME', 'PRODUCT_NAME'),
        3: ('111', 'in_relation', '48', 'sent1', 'sent1', 'BRAND_NAME', 'PRODUCT_NAME'),
        4: ('112', 'no_relation', '1', 'sent1', 'sent1', 'BRAND_NAME', ''),
        5: ('112', 'in_relation', '2', 'sent1', 'sent2', 'BRAND_NAME', 'PRODUCT_NAME'),
        6: ('112', 'in_relation', '2', 'sent3', 'sent4', 'BRAND_NAME', 'PRODUCT_NAME'),
        7: ('113', 'in_relation', '11', 'sent1', 'sent1', 'BRAND_NAME', 'PRODUCT_NAME'),
        8: ('113', 'in_relation', '12', 'sent1', 'sent1', 'BRAND_NAME', 'PRODUCT_NAME'),
        9: ('113', 'no_relation', '13', 'sent3', 'sent3', '', 'PRODUCT_NAME'),
        10: ('113', 'no_relation', '14', 'sent4', 'sent5', '', ''),
        11: ('113', 'no_relation', '15', 'sent4', 'sent5', '', ''),
        12: ('113', 'no_relation', '16', 'sent4', 'sent5', '', ''),
        13: ('113', 'no_relation', '17', 'sent4', 'sent5', '', ''),
        14: ('113', 'no_relation', '18', 'sent4', 'sent5', '', ''),
        15: ('113', 'no_relation', '19', 'sent4', 'sent5', '', ''),
        16: ('113', 'no_relation', '20', 'sent4', 'sent5', '', ''),
        17: ('113', 'no_relation', '21', 'sent4', 'sent5', '', ''),
        18: ('113', 'no_relation', '22', 'sent4', 'sent5', '', ''),
    }

    models = None

    dataset = Dataset(models, keys)
    sampler = Sampler(dataset)

    sampler.set_type = 'train'
    batch_gen = DataLoader(dataset, batch_size=3, sampler=sampler)
    print(list(batch_gen))

    sampler.set_type = 'valid'
    batch_gen = DataLoader(dataset, batch_size=3, sampler=sampler)
    print(list(batch_gen))

    sampler.set_type = 'test'
    batch_gen = DataLoader(dataset, batch_size=3, sampler=sampler)
    print(list(batch_gen))


if __name__ == "__main__":
    main()
