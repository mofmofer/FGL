# DataFrameを使う
import pandas as pd
# 描画
import matplotlib.pyplot as plt
# Feature Warningを無視する
from warnings import simplefilter
# GridSearch用のパラメタ探索範囲と評価指標に関する自作パッケージ
from function.param import *
# astropy
from astropy.table import Table
import astropy.units as u
from astropy.coordinates import SkyCoord

# Feature Warningを無視する
simplefilter(action='ignore', category=FutureWarning)

# シード値を設定しておく
np.random.seed(31415)

# プロット用のデータフレームを読み込む
FGL_plot = pd.read_csv("/Users/kawakami/Desktop/fight/STEP4/STEP4_5/data/FGL_unid_プロット用_閾値0.9.csv", header=0).drop(["Unnamed: 0"], axis=1)
print(FGL_plot.columns)
# 表示する画像サイズ
fig = plt.figure(figsize=(25, 10))
ax = fig.add_subplot(111, projection="aitoff")

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
    plot = ax.scatter(coord.l.wrap_at(180 * u.degree).radian, coord.b.radian, c=FGL_plot['proba_AGN'], cmap='coolwarm', s=5)

plt.colorbar(plot, ax=ax).ax.tick_params(labelsize=20)

ax.grid(True)

plt.savefig("/Users/kawakami/Desktop/fight/STEP4/STEP4_6/data/Predict.png", format='png', dpi=100)

# ラベルのカウントアップ
FGL_plot['label'].value_counts().to_csv("/Users/kawakami/Desktop/fight/STEP4/STEP4_6/data/予測クラス数.csv")
