# DataFrameを使う
import pandas as pd
import numpy as np

# 描画
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

# K-分割交差検証とパラメータフィッティング
from sklearn.model_selection import GridSearchCV, StratifiedKFold
# Feature Warningを無視する
from warnings import simplefilter
# DataFrameのデータの統計を確認する
import pandas_profiling as ppf

# 学習モデル
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
# GridSearch用のパラメタ探索範囲と評価指標に関する自作パッケージ
from function.param import *

# astropy
from astropy.table import Table
import astropy.units as u
from astropy.coordinates import SkyCoord

threshold = 0.9

# Feature Warningを無視する
simplefilter(action='ignore', category=FutureWarning)

# シード値を設定しておく
np.random.seed(31415)

# プロット用のデータフレームを読み込む
FGL_plot = pd.read_csv("/Users/kawakami/Desktop/fight/STEP1/STEP1_5/data/FGL_unid_プロット用_閾値{}.csv".format(threshold), header=0).drop(["Unnamed: 0"], axis=1)
FGL_plot = FGL_plot[FGL_plot['label'] == 'Otherlike']

# 表示する画像サイズ
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111, projection="aitoff")
plt.tick_params(labelsize=18)
# Astropy.Tableに変換
FGL_plot_astro = Table.from_pandas(FGL_plot)

# unit(degree)をGLON,GLAT列に追加
FGL_plot_astro['GLON'].unit = 'deg'
FGL_plot_astro['GLAT'].unit = 'deg'
print(FGL_plot_astro)

#  ソースクラスのリスト
source_classes = ['']
coord = SkyCoord(FGL_plot_astro["GLON"], FGL_plot_astro["GLAT"], frame="galactic")  # type: SkyCoord

# 描画
for source_class in source_classes:
    plot = ax.scatter(coord.l.wrap_at(180 * u.degree).radian, coord.b.radian, vmin=0, vmax=1, c=FGL_plot['proba_AGN']
               , cmap='coolwarm', s=200, marker="x")

# plt.colorbar(plot, ax=ax).ax.tick_params(labelsize=20)

ax.grid(True)
ax.legend([plot], ["otherlike"], loc=1, prop={'size': 23}, title_fontsize =23)
plt.savefig("/Users/kawakami/Desktop/fight/STEP1/STEP1_6/data/Predict_other_{}.png".format(threshold), format='png', dpi=100)
