from scipy.stats import chi2_contingency
from scipy.stats import chi2
import numpy as np
import pandas as pd

K1, K2 = map(int, input().split())
N = int(input())

X = []
Y = []
for i in range(N):
    f, s = map(int, input().split())
    X.append(f)
    Y.append(s)

df = pd.DataFrame(list(zip(X, Y)),
                  columns=['X', 'Y'])
crosstab = pd.crosstab(df['X'],
                            df['Y'],
                            margins=False)

print(crosstab)

stat, a1, a2, a3 = chi2_contingency(crosstab)

print(stat)
