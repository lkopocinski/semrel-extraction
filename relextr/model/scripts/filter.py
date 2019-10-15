
brands = set()
with open('valid.vectors') as in_file:
    for line in in_file:
        by_tab = line.split('\t')

        if len(by_tab) > 3:
            brands.add(by_tab[3])


with open('test.vectors') as in_file:
    for line in in_file:
        line = line.strip()
        by_tab = line.split('\t')

        if len(by_tab) > 3:
            if by_tab[3] not in brands:
                print(line)