import pandas as pd

root_path='../../generated'
paths = ['test', 'train', 'valid']
nrs = [81, 82, 83]

for path in paths:
    to_save = []
    for nr in nrs:
        df = pd.read_csv(f'{root_path}/{path}/positive/{nr}.context', sep='\t',
                         engine='python')
        brands = df.iloc[:,0]
        to_save.append(brands)

    hist = pd.concat(to_save)
    hist = hist.value_counts()
    hist.to_csv(f'{path}.pairs.hist', sep='\t', header=False)

