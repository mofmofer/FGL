"""
各データセットをcsvで出力し確認します。
解析用データの作成には必要ありません.
"""
# coding=utf-8

# DataFrameを使う
import pandas as pd
import numpy as np

# K-分割交差検証とパラメータフィッティング
# Feature Warningを無視する
from warnings import simplefilter
# DataFrameのデータの統計を確認する
import pandas_profiling as ppf
import sys
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Feature Warningを無視する
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)
# シード値を設定しておく
np.random.seed(31415)

# 最大表示行数の指定
pd.set_option('display.max_rows', None)

# csvデータを読み込みDataFrameに変換する
FGL = pd.read_csv("/Users/kawakami/Desktop/fight/STEP1/STEP1_2/4FGL_AGN_PSR_std.csv", header=0).drop(["Unnamed: 0"], axis=1)
# 読み込んだデータに対する統計プロファイルをHTMLに出力
ppf.ProfileReport(FGL).to_file(outputfile="/Users/kawakami/Desktop/fight/STEP1/STEP1_3/4FGL_AGN_PSR_std.html")
print(type(FGL.columns))
print(len(FGL.columns))


# DataFrame
X = FGL.drop("PSRness", axis=1)
y = FGL.loc[:, 'PSRness']

# 層化抽出法を用いたK-分割交差検証
skf = StratifiedKFold(n_splits=3, random_state=1)
skf.get_n_splits(X, y)

X_train_list, y_train_list, X_test_list, y_test_list = [], [], [], []
skf = StratifiedKFold(n_splits=3, random_state=1)
skf.get_n_splits(X, y)
for train_index, test_index in StratifiedKFold(n_splits=3).split(X, y):
    print(y.iloc[train_index].shape)
    print(y.iloc[train_index][lambda df: df == "Non-PSR"].shape)
    print(y.iloc[train_index][lambda df: df == "PSR"].shape)
    print(y.iloc[test_index].shape)
    print(y.iloc[test_index][lambda df: df == "Non-PSR"].shape)
    print(y.iloc[test_index][lambda df: df == "PSR"].shape)
    X_train_list.append(X.iloc[train_index])
    y_train_list.append(y.iloc[train_index])
    X_test_list.append(X.iloc[test_index])
    y_test_list.append(y.iloc[test_index])

sys.exit()

for i in range(len(X_train_list)):
    X_train_list[i].to_csv("/Users/kawakami/Desktop/fight/STEP1/STEP4_2/result/Xtrain{}.csv".format(i+1))
    y_train_list[i].to_csv("/Users/kawakami/Desktop/fight/STEP1/STEP4_2/result/ytrain{}.csv".format(i+1))
    X_test_list[i].to_csv("/Users/kawakami/Desktop/fight/STEP1/STEP4_2/result/Xtest{}.csv".format(i+1))
    y_test_list[i].to_csv("/Users/kawakami/Desktop/fight/STEP1/STEP4_2/result/ytest{}.csv".format(i+1))
