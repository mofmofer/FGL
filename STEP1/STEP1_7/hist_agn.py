# Gooness of fit test 適合度検定に使うライブラリ
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare, chi2_contingency

# 銀緯のbin分け
bins = np.arange(-180, 181, 4).tolist()

FGL_AGNness = pd.read_csv("/Users/kawakami/Desktop/fight/STEP1/STEP1_1/4FGL_AGN.csv", header=0).drop(["Unnamed: 0"], axis=1)['GLON']
FGL_AGNlike = pd.read_csv("/Users/kawakami/Desktop/fight/STEP1/STEP1_5/data/FGL_unid_AGN_like_閾値0.9.csv", header=0).drop(["Unnamed: 0"], axis=1)['GLON']

for i in range(len(FGL_AGNness)):
    if FGL_AGNness[i] > 180:
        FGL_AGNness[i] = FGL_AGNness[i]-360
for i in range(len(FGL_AGNlike)):
    if FGL_AGNlike[i] > 180:
        FGL_AGNlike[i] = FGL_AGNlike[i]-360

FGL_cut_AGNness = pd.cut(FGL_AGNness, bins).value_counts().sort_index().tolist()
FGL_cut_AGNlike = pd.cut(FGL_AGNlike, bins).value_counts().sort_index().tolist()
# 規格化前
print(FGL_cut_AGNness)
print(sum(FGL_cut_AGNlike), sum(FGL_cut_AGNness))

ratio = sum(FGL_cut_AGNlike)/sum(FGL_cut_AGNness)
FGL_cut_AGNness = [i*ratio for i in FGL_cut_AGNness]
# 規格化後
FGL_cut_AGNness = np.round(FGL_cut_AGNness, decimals=1)
print(FGL_cut_AGNness)
print(FGL_cut_AGNlike)
print(sum(FGL_cut_AGNlike), sum(FGL_cut_AGNness))

# Pythonを利用して適合度検定
print('適合度検定',chisquare(FGL_cut_AGNlike, f_exp=FGL_cut_AGNness))

# Pythonを利用して独立性の検定
x2, p, dof, e = chi2_contingency([FGL_cut_AGNlike,FGL_cut_AGNness])
print(f'p値 　　　= {p :.3f}')
print(f'カイ2乗値 = {x2:.2f}')
print(f'自由度　  = {dof}')

plt.figure(figsize=(60, 11))
plt.hist(FGL_AGNness, alpha=0.5, bins=90, label="AGNness", color='#4A593D')
plt.hist(FGL_AGNlike, bins=90, label='AGNlike', color='#B5CAA0')
plt.legend(loc="upper left", fontsize=40)
plt.tick_params(labelsize=40)
plt.xlim(-180, 180)
plt.xlabel('GLAT',fontsize=40)
plt.ylabel('Count',fontsize=40)
plt.savefig("/Users/kawakami/Desktop/fight/STEP1/STEP1_7/hist_agnness.png", format='png', dpi=100)



