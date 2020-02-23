# DataFrameを使う
import pandas as pd
# 描画
import matplotlib.pyplot as plt
# Feature Warningを無視する
from warnings import simplefilter
# GridSearch用のパラメタ探索範囲と評価指標に関する自作パッケージ
from function.param import *

threshold = 0.9

# Feature Warningを無視する
simplefilter(action='ignore', category=FutureWarning)
# シード値を設定しておく
np.random.seed(31415)

# プロット用のデータフレームを読み込む
FGL_plot = pd.read_csv("/Users/kawakami/Desktop/fight/STEP5/STEP5_5/data_SVM/FGL_unid_プロット用_閾値{}.csv".format(threshold), header=0).drop(["Unnamed: 0"], axis=1)
FGL_plot1 = FGL_plot[FGL_plot['label'] == 'PSRlike']['GLON'].values.tolist()
FGL_plot2 = FGL_plot[FGL_plot['label'] == 'AGNlike']['GLON'].values.tolist()
FGL_plot3 = FGL_plot[FGL_plot['label'] == 'Otherlike']['GLON'].values.tolist()

for i in range(len(FGL_plot1)):
    if FGL_plot1[i] > 180:
        FGL_plot1[i] = FGL_plot1[i]-360
for i in range(len(FGL_plot2)):
    if FGL_plot2[i] > 180:
        FGL_plot2[i] = FGL_plot2[i]-360
for i in range(len(FGL_plot3)):
    if FGL_plot3[i] > 180:
        FGL_plot3[i]=FGL_plot3[i]-360

plt.figure(figsize=(50, 10))
plt.hist(FGL_plot2, alpha=0.5, bins=90, label="AGNlike", color='crimson')
plt.hist(FGL_plot1, bins=90, label='PSRlike')
plt.legend(loc="upper left", fontsize=40)
plt.tick_params(labelsize=40)
plt.xlim(-180, 180)
plt.xlabel('天体数',fontsize=18)
plt.ylabel('銀緯',fontsize=18)
plt.savefig("/Users/kawakami/Desktop/fight/STEP5/STEP5_6/SVM/data/hist_agn_and_psr.png", format='png', dpi=100)

plt.figure(figsize=(50, 10))
plt.hist(FGL_plot3, alpha=0.5, bins=90, label="Others", color='gray')
plt.legend(loc="upper left", fontsize=40)
plt.tick_params(labelsize=40)
plt.xlim(-180, 180)
plt.xlabel('天体数',fontsize=18)
plt.ylabel('銀緯',fontsize=18)
plt.savefig("/Users/kawakami/Desktop/fight/STEP5/STEP5_6/SVM/data/hist_other.png", format='png', dpi=100)
