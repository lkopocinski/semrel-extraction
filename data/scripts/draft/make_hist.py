import pandas as pd

root_path = '../../generations'
paths = ['test', 'train', 'valid']
nrs = [81, 82, 83]

for nr in nrs:
    to_save = []
    for path in paths:
        df = pd.read_csv(f'{root_path}/{path}/positive/{nr}.context', sep='\t| : ', engine='python')
        brands = df[df.iloc[:,2] == "BRAND_NAME"].iloc[:,0]
        to_save.append(brands)

        brands = df[df.iloc[:,3] == "BRAND_NAME"].iloc[:,1]
        to_save.append(brands)

    hist = pd.concat(to_save)
    hist = hist.value_counts()
    hist.to_csv(f'{nr}.hist', sep='\t', header=False)
