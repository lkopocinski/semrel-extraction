brands = set()
with open('../../relextr/model/dataset/valid.vectors') as in_file:
    for line in in_file:
        by_tab = line.split('\t')

        if len(by_tab) > 3:
            brands.add(by_tab[3])

with open('../../relextr/model/dataset/train.vectors') as in_file:
    for line in in_file:
        by_tab = line.split('\t')

        if len(by_tab) > 3:
            brands.add(by_tab[3])

with open('../../relextr/model/dataset/test.vectors') as in_file:
    for line in in_file:
        line = line.strip()
        by_tab = line.split('\t')

        if len(by_tab) > 3:
            if by_tab[3] not in brands:
                print(line)
