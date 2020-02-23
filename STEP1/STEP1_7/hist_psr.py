# Gooness of fit test 適合度検定に使うライブラリ
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare, chi2_contingency

# 銀緯のbin分け
bins = np.arange(-180, 181, 20).tolist()

FGL_PSRness = pd.read_csv("/Users/kawakami/Desktop/fight/STEP1/STEP1_1/4FGL_PSR.csv", header=0).drop(["Unnamed: 0"], axis=1)['GLON']
FGL_PSRlike = pd.read_csv("/Users/kawakami/Desktop/fight/STEP1/STEP1_5/data/FGL_unid_PSR_like_閾値0.9.csv", header=0).drop(["Unnamed: 0"], axis=1)['GLON']

for i in range(len(FGL_PSRness)):
    if FGL_PSRness[i] > 180:
        FGL_PSRness[i] = FGL_PSRness[i] - 360
for i in range(len(FGL_PSRlike)):
    if FGL_PSRlike[i] > 180:
        FGL_PSRlike[i] = FGL_PSRlike[i] - 360

FGL_cut_PSRness = pd.cut(FGL_PSRness, bins).value_counts().sort_index().tolist()
FGL_cut_PSRlike = pd.cut(FGL_PSRlike, bins).value_counts().sort_index().tolist()
# 規格化前
print(FGL_cut_PSRness)
print(sum(FGL_cut_PSRlike), sum(FGL_cut_PSRness))

ratio = sum(FGL_cut_PSRlike) / sum(FGL_cut_PSRness)
FGL_cut_PSRness = [i * ratio for i in FGL_cut_PSRness]
# 規格化後
FGL_cut_PSRness = np.round(FGL_cut_PSRness, decimals=1)
print(FGL_cut_PSRness)
print(FGL_cut_PSRlike)
print(sum(FGL_cut_PSRlike), sum(FGL_cut_PSRness))

# Pythonを利用して適合度検定
print('適合度検定', chisquare(FGL_cut_PSRlike, f_exp=FGL_cut_PSRness))

# Pythonを利用して独立性の検定
x2, p, dof, e = chi2_contingency([FGL_cut_PSRlike, FGL_cut_PSRness])
print(f'p値 　　　= {p :.3f}')
print(f'カイ2乗値 = {x2:.2f}')
print(f'自由度　  = {dof}')

plt.figure(figsize=(60, 11))
plt.hist(FGL_PSRness, alpha=0.5, bins=90, label="PSRness", color='#FEDFE1')
plt.hist(FGL_PSRlike, bins=90, label='PSRlike', color='#DB4D6D')
plt.legend(loc="upper left", fontsize=40)
plt.tick_params(labelsize=40)
plt.xlim(-180, 180)
plt.xlabel('GLAT',fontsize=40)
plt.ylabel('Count',fontsize=40)
plt.savefig("/Users/kawakami/Desktop/fight/STEP1/STEP1_7/hist_psrness.png", format='png', dpi=400)



