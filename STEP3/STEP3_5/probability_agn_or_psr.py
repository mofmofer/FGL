# coding=utf-8

# DataFrameを使う
import pandas as pd

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

threshold = 0.9

# Feature Warningを無視する
simplefilter(action='ignore', category=FutureWarning)
# シード値を設定しておく
np.random.seed(31415)

# csvデータを読み込みDataFrameに変換する
FGL = pd.read_csv("/Users/kawakami/Desktop/fight/STEP3/STEP3_2/4FGL_AGN_PSR_std.csv", header=0).drop(["Unnamed: 0"], axis=1)
FGL_unid = pd.read_csv("/Users/kawakami/Desktop/fight/STEP3/STEP3_2/4FGL_Unid_std.csv", header=0).drop(["Unnamed: 0"], axis=1)
"""
STEP3_2で生成されたデータは位置情報Dropしてる.完全な対応を持つSTEP1_2のプロット用データを代用.
"""
FGL_plot = pd.read_csv("/Users/kawakami/Desktop/fight/STEP1/STEP1_2/4FGL_Unid_plot.csv", header=0).drop(["Unnamed: 0"], axis=1)

# 読み込んだデータに対する統計プロファイルをHTMLに出力
ppf.ProfileReport(FGL_unid).to_file(outputfile="/Users/kawakami/Desktop/fight/STEP3/STEP5_5/data/unid.html")

# DataFrame
X = FGL.drop("PSRness", axis=1)
y = FGL.PSRness

X_unid = FGL_unid.drop("PSRness", axis=1)

# 層化抽出法を用いたK-分割交差検証
skf = StratifiedKFold(n_splits=3, random_state=1)
skf.get_n_splits(X, y)
print("k-foldの設定を確認します : ", skf)

# DataFrame -> np.array

print("データのサイズを確認します")
print("X: ", X.shape)

X = X.values
y = pd.get_dummies(y)['PSR']

# 勾配ブースティング ----------------------------

grb = GradientBoostingClassifier(max_depth=7, max_features='sqrt', min_samples_leaf=90, min_samples_split=4,
                                 subsample=0.8)
grb.fit(X, y)
# 予測結果の保存
np.savetxt('/Users/kawakami/Desktop/fight/STEP3/STEP5_5/data/unid_prob_{}.csv'.format(threshold), grb.predict_proba(X_unid), delimiter=',')

# プロット用のデータフレームに分類確率のデータを追加、
FGL_plot = FGL_plot.loc[:, ['GLON', 'GLAT']]
FGL_plot['proba_AGN'] = grb.predict_proba(X_unid)[:, 0]
FGL_plot['proba_PSR'] = grb.predict_proba(X_unid)[:, 1]

# プロット用のデータフレームに分類確率によるラベルを追加
label = []
for i in grb.predict_proba(X_unid)[:, 0]:
    if i > threshold:
        label.append("AGNlike")
    elif i > 1-threshold:
        label.append("Otherlike")
    else:
        label.append("PSRlike")
FGL_plot['label'] = label

# プロット用のデータフレームをcsv形式で保存
FGL_plot.to_csv("/Users/kawakami/Desktop/fight/STEP3/STEP5_5/data/FGL_unid_プロット用_閾値{}.csv".format(threshold))
FGL_plot[FGL_plot['label'] == 'AGNlike'].to_csv("/Users/kawakami/Desktop/fight/STEP3/STEP5_5/data/FGL_unid_AGN_like_閾値{}.csv".format(threshold))
FGL_plot[FGL_plot['label'] == 'PSRlike'].to_csv("/Users/kawakami/Desktop/fight/STEP3/STEP5_5/data/FGL_unid_PSR_like_閾値{}.csv".format(threshold))
FGL_plot[FGL_plot['label'] == 'Otherlike'].to_csv("/Users/kawakami/Desktop/fight/STEP3/STEP5_5/data/FGL_unid_Other_like_閾値{}.csv".format(threshold))
