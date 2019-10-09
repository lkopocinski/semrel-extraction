import pandas as pd

root='../generated'
paths = ['test', 'train', 'valid']
nrs = [81, 82, 83]

for path in paths:
    to_save = []
    for nr in nrs:
        df = pd.read_csv(f'{root}/{path}/positive/{nr}.context', sep='\t| : ')
        brands = df.iloc[:,0]
        to_save.append(brands)
    
    hist = pd.concat(to_save)
    hist = hist.value_counts()
    hist.to_csv(f'{path}.hist', sep='\t')
        


