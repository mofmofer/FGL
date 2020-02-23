"""
STEP3-1で出力されたデータに対し欠損値・標準化処理を行います。
4FGL.csv : fitsの生データ
4FGL_AGN_PSR : 特徴量"CLASS1"をもとにAGNとPSRだけを抽出したデータ(特徴量+出力値)
4FGL_Unid : 特徴量"CLASS1"をもとにUnidだけを抽出したデータ(特徴量+出力値)

STEP3-1とSTEP3-2で解析に必要なデータの準備を完了できます.
"""
# coding=utf-8
import pandas as pd
from sklearn.preprocessing import StandardScaler

FGL_AGN_PSR = pd.read_csv("/Users/kawakami/Desktop/fight/STEP5/STEP5_1/4FGL_AGN_PSR.csv", header=0).drop(["Unnamed: 0"],
                                                                            axis=1)
FGL_AGN_PSR['Extended_Source_Name'] = FGL_AGN_PSR['Extended_Source_Name'].astype(object)
FGL_Unid = pd.read_csv("/Users/kawakami/Desktop/fight/STEP5/STEP5_1/4FGL_Unid.csv", header=0).drop(["Unnamed: 0"], axis=1)
FGL_Unid['Extended_Source_Name'] = FGL_Unid['Extended_Source_Name'].astype(object)

# 各列のデータ型を確認
# print(FGL_AGN_PSR.dtypes)

# 標準化対象の(数値の)列を抽出
AGN_PSR_values = FGL_AGN_PSR.select_dtypes(exclude='object')
Unid_values = FGL_Unid.select_dtypes(exclude='object')
# 標準化各列の標準偏差を確認(標準偏差0で0徐算が起きるので注意)
# print(AGN_PSR_values.STEP4_3(numeric_only=True).head())

# Unidプロット用
FGL_Unid.to_csv("/Users/kawakami/Desktop/fight/STEP5/STEP5_2/4FGL_Unid_plot.csv")

# データを標準化する
sts = StandardScaler()
# print(standard.STEP4_3(numeric_only=True).head())
standard = pd.DataFrame(sts.fit_transform(AGN_PSR_values), columns=AGN_PSR_values.columns)
# print(standard.STEP4_3(numeric_only=True).head())
standard['PSRness'] = FGL_AGN_PSR.PSRness
print(standard.columns.shape)
standard.to_csv("/Users/kawakami/Desktop/fight/STEP5/STEP5_2/4FGL_AGN_PSR_std.csv")

standard = pd.DataFrame(sts.fit_transform(Unid_values), columns=Unid_values.columns)
standard['PSRness'] = FGL_Unid.PSRness
standard.to_csv("/Users/kawakami/Desktop/fight/STEP5/STEP5_2/4FGL_Unid_std.csv")


