"""
4FGLカタログに対しAGN、PSRのラベルづけを行います.
DataFrameとして処理したのち、csvファイルに出力します.
"""

# coding=utf-8
from astropy.table import Table
import pandas as pd
from time import sleep
from tqdm import tqdm

# 最大表示行数の指定
pd.set_option('display.max_rows', None)

# 4FGLソースリストの読み込み
filename = '/Users/kawakami/Desktop/fight/gll_psc_v19.fit'
table = Table.read(filename, hdu=1)

# Fluxデータの分割
flux_range = ['50_100', '100_300', '300_1000', '1000_3000', '3000_10000', '10000_30000', '30000_100000']
for i in range(len(flux_range)):
    new_column = 'Flux' + flux_range[i]
    table[new_column] = table['Flux_Band'][:, i].astype('float64')

# 拡散源データの取得
extended_source_table = Table.read(str(filename), hdu='ExtendedSources')

# 4FGLの特徴量名のタプル(73要素)
column_names = tuple(name for name in table.colnames if len(table[name].shape) <= 1)
# データ型をAstropy.Tableからpandas.Dataframeへ変換
FGL = table[column_names].to_pandas()

# 欠損値
df_missing = pd.DataFrame()  # 欠損値保存用DataFrame
missing1 = FGL.isnull().sum()
missing1.name = '4FGL_init'
df_missing = df_missing.append(missing1)
df_missing.loc[:, 'AGNness'] = None
df_missing.loc[:, 'PSRness'] = None

# byte型の文字列をStringに変換
print('byte型の文字列をString型に変更します')
sleep(1)
for i in tqdm(range(len(FGL['CLASS1']))):
    FGL.loc[i, 'CLASS1'] = FGL.loc[i, 'CLASS1'].decode('utf-8').strip()
    if FGL.loc[i, 'CLASS1'] == '':
        FGL.loc[i, 'CLASS1'] = 'unid'
for i in tqdm(range(len(FGL['Source_Name']))):
    FGL.loc[i, 'Source_Name'] = FGL.loc[i, 'Source_Name'].decode('utf-8').strip()
for i in tqdm(range(len(FGL['ASSOC1']))):
    FGL.loc[i, 'ASSOC1'] = FGL.loc[i, 'ASSOC1'].decode('utf-8').strip()
for i in tqdm(range(len(FGL['Extended_Source_Name']))):
    FGL.loc[i, 'Extended_Source_Name'] = FGL.loc[i, 'Extended_Source_Name'].decode('utf-8').strip()
for i in tqdm(range(len(FGL['SpectrumType']))):
    FGL.loc[i, 'SpectrumType'] = FGL.loc[i, 'SpectrumType'].decode('utf-8').strip()
sleep(1)

# AGNのラベルづけ
print('AGNのラベルづけを行っています...')
agn_class = ['BCU', 'bcu', 'BLL', 'bll', 'FSRQ', 'fsrq', 'RDG', 'rdg', 'NLSY1', 'nlsy1', 'agn', 'ssrq', 'sey']
FGL['AGNness'] = ''
sleep(1)
for i in tqdm(range(len(FGL['CLASS1']))):
    if FGL.loc[i, 'CLASS1'] in agn_class:
        FGL.loc[i, 'AGNness'] = 'AGN'
    else:
        FGL.loc[i, 'AGNness'] = 'Non-AGN'
sleep(1)
# ラベルカウント
FGL_AGN = FGL['AGNness'] == 'AGN'
FGL_nonAGN = FGL['AGNness'] == 'Non-AGN'
print('AGNでラベルづけされた天体数:{0}、その他天体数{1}'.format(FGL_AGN.sum(), FGL_nonAGN.sum()), '\n')

# PSRのラベルづけ
print('PSRのラベルづけを行っています...')
psr_class = ['PSR', 'psr']
FGL['PSRness'] = ''
sleep(1)
for i in tqdm(range(len(FGL['CLASS1']))):
    if FGL.loc[i, 'CLASS1'] in psr_class:
        FGL.loc[i, 'PSRness'] = 'PSR'
    else:
        FGL.loc[i, 'PSRness'] = 'Non-PSR'
sleep(1)
FGL_PSR = FGL['PSRness'] == 'PSR'
FGL_nonPSR = FGL['PSRness'] == 'Non-PSR'
print('PSRでラベルづけされた天体数:{0}、その他天体{1}'.format(FGL_PSR.sum(), FGL_nonPSR.sum()), '\n')
FGL = FGL.drop(
    ['RAJ2000', 'DEJ2000', 'Conf_95_SemiMajor', 'Conf_95_SemiMinor', 'Flux100_300', 'Conf_68_PosAng', 'Unc_PLEC_Exp_Index', 'ASSOC2',
     'Signif_Peak', 'Flux_Peak', 'Unc_Flux_Peak', 'Time_Peak', 'Peak_Interval', 'Unc_LP_beta',
     'ASSOC_FGL', 'ASSOC_FHL', 'ASSOC_GAM1', 'ASSOC_GAM2', 'ASSOC_GAM3', 'ASSOC_PROB_BAY', 'ASSOC_PROB_LR',
     'TEVCAT_FLAG', 'ASSOC_TEV', 'Unc_Counterpart', 'RA_Counterpart', 'DEC_Counterpart', 'ASSOC1', 'CLASS2'], axis=1)
# 欠損値データ
df_missing.loc['drop'] = None
missing2 = FGL.isnull().sum()
for i in range(missing2.size):
    df_missing.loc['drop', missing2.index[i]] = missing2[i]
# AGN&PSR と Unid のDataFrameを作成
FGL_AGN_PSR = FGL[lambda df: (df.PSRness == 'PSR') | (df.AGNness == 'AGN')]
# FGL_unid = FGL[lambda df: (df.PSRness == 'Non-PSR') & (df.AGNness == 'Non-AGN')]
FGL_unid = FGL[lambda df: (df.CLASS1 == 'unid')]
print(FGL_unid.shape)

# 欠損値の処理
print('欠損値の処理を行います')
print('AGN&PSR:{}, UNID:{} ====> '.format(FGL_AGN_PSR.shape, FGL_unid.shape), end="")
FGL_AGN_PSR = FGL_AGN_PSR.dropna()
FGL_unid = FGL_unid.dropna()
print('AGN&PSR:{}, UNID:{}'.format(FGL_AGN_PSR.shape, FGL_unid.shape))

FGL_AGN_PSR[lambda df: (df.PSRness == "PSR")].loc[:, ["GLON", "GLAT", "PSRness"]].to_csv("/Users/kawakami/Desktop/fight/STEP5/STEP5_1/4FGL_PSR.csv")
FGL_AGN_PSR[lambda df: (df.PSRness == "Non-PSR")].loc[:, ["GLON", "GLAT", "PSRness"]].to_csv("/Users/kawakami/Desktop/fight/STEP5/STEP5_1/4FGL_AGN.csv")
# 天体数の確認
print('PSR:{} AGN:{} PSR+AGN:{} Unid :{}'.format(len(FGL_AGN_PSR[FGL_AGN_PSR.loc[:, 'PSRness'] == 'PSR']),
                                                 len(FGL_AGN_PSR[FGL_AGN_PSR.loc[:, 'PSRness'] == 'Non-PSR']),
                                                 len(FGL_AGN_PSR),
                                                 len(FGL_unid)))
