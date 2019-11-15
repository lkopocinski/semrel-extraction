brands = set()
with open('../../../relextr/model/dataset/valid.vectors') as in_file:
    for line in in_file:
        by_tab = line.split('\t')

        if len(by_tab) > 3:
            if by_tab[4] == 'BRAND_NAME':
                brands.add(by_tab[3])
            elif by_tab[7] == 'BRAND_NAME':
                brands.add(by_tab[6])


with open('../../../relextr/model/dataset/train.vectors') as in_file:
    for line in in_file:
        by_tab = line.split('\t')

        if len(by_tab) > 3:
            if by_tab[4] == 'BRAND_NAME':
                brands.add(by_tab[3])
            elif by_tab[7] == 'BRAND_NAME':
                brands.add(by_tab[6])

print('\n', brands)

with open('../../../relextr/model/dataset/test.vectors') as in_file:
    for line in in_file:
        line = line.strip()
        by_tab = line.split('\t')

        if len(by_tab) > 3:
#            if by_tab[3] not in brands:
#                print(line)

            if by_tab[4] == 'BRAND_NAME':
                if by_tab[3] in brands:
                    print(by_tab[3:5], by_tab[6:8])
            elif by_tab[7] == 'BRAND_NAME':
                if by_tab[6] in brands:
                    print(by_tab[3:5], by_tab[6:8])


