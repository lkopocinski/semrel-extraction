#!/usr/bin/env python
# -*- coding: utf-8 -*-
from data_loader import BrandProductDataset, DatasetGenerator
from torch.utils.data import DataLoader


def main():
    keys = {
        0: ('111', 'in_relation', '17', 'sent1', 'sent1', 'BRAND_NAME', 'PRODUCT_NAME', 'Huawei', 'P8'),
        1: ('111', 'no_relation', '23', 'sent1', 'sent2', 'BRAND_NAME', 'PRODUCT_NAME', 'Huawei', 'S9'),
        2: ('111', 'in_relation', '48', 'sent1', 'sent1', 'BRAND_NAME', 'PRODUCT_NAME', 'Samsung', 'S8'),
        3: ('112', 'no_relation', '1', 'sent1', 'sent1', 'BRAND_NAME', '', 'Samsung', 'telefon'),
        4: ('112', 'in_relation', '2', 'sent1', 'sent2', 'BRAND_NAME', 'PRODUCT_NAME', 'Nivea', 'Soft'),
        5: ('112', 'in_relation', '2', 'sent3', 'sent4', 'BRAND_NAME', 'PRODUCT_NAME', 'Nivea', 'For Men'),
        6: ('113', 'in_relation', '11', 'sent1', 'sent1', 'BRAND_NAME', 'PRODUCT_NAME', 'Konto', 'Plus50'),
        7: ('113', 'in_relation', '12', 'sent1', 'sent1', 'BRAND_NAME', 'PRODUCT_NAME', 'SHARP', 'AQUOS'),
        8: ('113', 'no_relation', '13', 'sent3', 'sent3', '', 'PRODUCT_NAME', 'dom', 'WRT-300'),
        9: ('113', 'no_relation', '14', 'sent4', 'sent5', '', '', 'dom', 'szafka'),
        10: ('113', 'no_relation', '15', 'sent4', 'sent5', '', '', 'telewizor', 'podstawa'),
        11: ('113', 'no_relation', '16', 'sent4', 'sent5', '', '', 'telewizor', 'ekran'),
        12: ('113', 'no_relation', '17', 'sent4', 'sent5', '', '', 'smartfon', 'kabel'),
        13: ('113', 'no_relation', '18', 'sent4', 'sent5', '', '', 'instrukcja', 'telewizja'),
        14: ('113', 'no_relation', '19', 'sent4', 'sent5', '', '', 'jutro', 'artykuł'),
        15: ('113', 'no_relation', '20', 'sent4', 'sent5', '', '', 'chemia', 'perfuma'),
        16: ('113', 'no_relation', '21', 'sent4', 'sent5', '', '', 'herbata', 'kubek'),
        17: ('113', 'no_relation', '22', 'sent4', 'sent5', '', '', 'ekran', 'piksel'),
    }

    models = None

    dataset = BrandProductDataset(models, keys)
    sampler = DatasetGenerator(dataset)

    sampler.set_type = 'train'
    batch_gen = DataLoader(dataset, batch_size=2, sampler=sampler)
    print(list(batch_gen))

    sampler.set_type = 'valid'
    batch_gen = DataLoader(dataset, batch_size=3, sampler=sampler)
    print(list(batch_gen))

    sampler.set_type = 'test'
    batch_gen = DataLoader(dataset, batch_size=3, sampler=sampler)
    print(list(batch_gen))


if __name__ == "__main__":
    main()
