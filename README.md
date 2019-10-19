# semrel extraction
A project focused on mining semantic relations.working tree
## Experiments results
| experiment | loss | accuracy | precision | recall | fscore |
|------------|------|----------|-----------|--------|--------|
|add fast text vector|258.5631501157622 |0.9113468489706124 |0.9544711614018546, 0.6706270627062704 |0.9336345539315838, 0.7127062706270627 |0.9403705409268975, 0.6663361574252662|
|add hard negative|181.68987370255036 |0.8723348525328747 |0.9398527948032901, 0.5934543454345433 |0.9003876578134006, 0.6627612761276127 |0.9141865265298127, 0.5915003405102419|
|dvc|191.7700411012629 |0.8616061606160637 |0.944504688564095, 0.5873801665880873 |0.8806197286395304, 0.708195819581958 |0.9061453422188187, 0.6104898585096605|
|elmo conv only|93.97827923670411 |0.8644179894179894 |0.94638901 0.6232073  |0.88082474 0.79799666 |0.91243059 0.69985359|
|elmo conv vector|88.0158392097801 |0.8839285714285714 |0.95165505 0.67032967 |0.90103093 0.81469115 |0.92565135 0.73549359|
|elmo convolution embedings|25.04905278934166 |0.88470066518847 |0.95110024 0.23809524 |0.9239905  0.33333333 |0.9373494  0.27777778|
|extract layer|89.51170005346648 |0.878968253968254 |0.95015304 0.65807327 |0.89608247 0.8096828  |0.92232598 0.7260479 |
|fix substitution|76.56391648761928 |0.9001652892561983 |0.92684887 0.77653631 |0.95053586 0.69616027 |0.93854294 0.73415493|
|fix substitution conv|79.89077579509467 |0.8922314049586777 |0.91699762 0.76923077 |0.95177246 0.65108514 |0.93406149 0.70524412|
|fscore stoping|209.95529462682254 |0.8729949185394753 |0.9449931897951701, 0.5792629262926292 |0.8950298601288701, 0.6801980198019801 |0.9139890568798094, 0.5950175969977958|
|lexical split|96.85194670106284 |0.8756613756613757 |0.94601654 0.65337001 |0.89608247 0.79298831 |0.92037272 0.71644042|
|more epoch|191.35016771091614 |0.8592959295929612 |0.9476124993451731, 0.5910655351249412 |0.8755631515532505, 0.7223322332233222 |0.9043985042988821, 0.6172724415298672|
|only sent2vec lexical|39.7109414935112 |0.9057649667405765 |0.93356243 0.06896552 |0.96793349 0.03333333 |0.95043732 0.04494382|
|pos embeddings|25.04905278934166 |0.88470066518847 |0.95110024 0.23809524 |0.9239905  0.33333333 |0.9373494  0.27777778|
|sent2vec|84.80066628009081 |0.8898809523809523 |0.9340249  0.71661238 |0.92824742 0.7345576  |0.9311272  0.72547403|
|sent2vec mask two|84.80066628009081 |0.8898809523809523 |0.9340249  0.71661238 |0.92824742 0.7345576  |0.9311272  0.72547403|
|sentence window|96.85194670106284 |0.8756613756613757 |0.94601654 0.65337001 |0.89608247 0.79298831 |0.92037272 0.71644042|
|statistics|89.51170005346648 |0.878968253968254 |0.95015304 0.65807327 |0.89608247 0.8096828  |0.92232598 0.7260479 |
|test lexical|25.04905278934166 |0.88470066518847 |0.95110024 0.23809524 |0.9239905  0.33333333 |0.9373494  0.27777778|
|test lexical sent2vec|26.241446510422975 |0.8869179600886918 |0.95121951 0.24390244 |0.9263658  0.33333333 |0.93862816 0.28169014|
|word2vec vectors|100.69416188262403 |0.8723544973544973 |0.94229935 0.64812239 |0.8956701  0.77796327 |0.91839323 0.70713202|
