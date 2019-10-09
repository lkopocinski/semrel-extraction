import pandas as pd

hist_file = 'test.pairs.hist'
pred_file = 'test.prediction_'

hist = pd.read_csv(hist_file, sep='\t', names=['pair', 'freq'])
#print(hist.head())

pred = pd.read_csv(pred_file, sep='\t')
pred = pred.dropna()
pred = pred.iloc[:, 0:3]
pred.columns = ['pair', 'pred', 'true']
#print(pred.head())

hist_pred = pd.merge(pred, hist, how='right', on=['pair'])
hist_pred = hist_pred[hist_pred['true'] == 'true: in_relation']
hist_pred = hist_pred[hist_pred['pred'] == 'pred: in_relation']
#print(hist_pred.head())

positive_pred = hist_pred.groupby('pair')['pred'].value_counts()
print(positive_pred.head())
#positive_pred = positive_pred.reset_index()
#print(positive_pred.head())


results = pd.merge(positive_pred, hist_pred, how='left', on=['pair'])

#print(results.head(10))

