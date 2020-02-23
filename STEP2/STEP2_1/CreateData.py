# coding=utf-8
from astropy.table import Table
import numpy as np
from scipy import stats


# SEDからFluxに変換
def sed_to_flux(photon_flux, alpha, Elo, Ehi):
    R = Elo / Ehi
    GeVToerg = 0.00160217657
    sedFlux = (GeVToerg * Elo) * (alpha - 1) * photon_flux * (pow(R, (alpha / 2 - 1)) / (1 - pow(R, (alpha - 1))))
    return sedFlux


# 4FGLソースリストの読み込み
filename = '/Users/kawakami/Desktop/fight/gll_psc_v19.fit'
table = Table.read(filename, hdu=1)

# Fluxデータの分割
Flux = ['Flux50_100', 'Flux100_300', 'Flux300_1000', 'Flux1000_3000', 'Flux3000_10000', 'Flux10000_30000',
        'Flux30000_100000']
for i in range(len(Flux)):
    table[Flux[i]] = table['Flux_Band'][:, i].astype('float64')

extended_source_table = Table.read(str(filename), hdu='ExtendedSources')
scalar_column_names = tuple(name for name in table.colnames if len(table[name].shape) <= 1)

FGL = table[scalar_column_names].to_pandas()
FGL.to_csv('/Users/kawakami/Desktop/fight/STEP2/STEP2_1/4FGL.csv')

FGL = FGL.loc[:,
      ['Source_Name', 'RAJ2000', 'DEJ2000', 'GLON', 'GLAT',
       'PL_Index', 'LP_Index', 'PLEC_Index',
       'Energy_Flux100', 'Variability_Index',
       'Conf_68_SemiMajor', 'Conf_68_SemiMinor', 'Conf_68_PosAng', 'Conf_95_SemiMajor', 'Conf_95_SemiMinor',
       'Conf_95_PosAng',
       'Signif_Avg', 'Pivot_Energy',
       'PL_Flux_Density', 'LP_Flux_Density', 'PLEC_Flux_Density',
       'PL_Unc_Flux_Density', 'LP_Unc_Flux_Density', 'PLEC_Unc_Flux_Density',
       'Flux1000', 'Unc_Flux1000', 'Energy_Flux100', 'Unc_Energy_Flux100',
       'LP_SigCurv', 'PLEC_SigCurv',
       'Flux100_300', 'Flux300_1000', 'Flux1000_3000', 'Flux3000_10000', 'Flux10000_30000', 'Flux30000_100000',
       'CLASS1', 'ASSOC1']]

FGL = FGL.assign(PL_SED100_300=lambda df: sed_to_flux(df.Flux100_300, df.PL_Index, 0.1, 0.3),
                 PL_SED300_1000=lambda df: sed_to_flux(df.Flux300_1000, df.PL_Index, 0.3, 1.0),
                 PL_SED1000_3000=lambda df: sed_to_flux(df.Flux1000_3000, df.PL_Index, 1.0, 3.0),
                 PL_SED3000_10000=lambda df: sed_to_flux(df.Flux3000_10000, df.PL_Index, 3.0, 10.0),
                 PL_SED10000_30000=lambda df: sed_to_flux(df.Flux10000_30000, df.PL_Index, 10.0, 30.0),
                 PL_SED30000_100000=lambda df: sed_to_flux(df.Flux30000_100000, df.PL_Index, 30.0, 100.0),
                 LP_SED100_300=lambda df: sed_to_flux(df.Flux100_300, df.LP_Index, 0.1, 0.3),
                 LP_SED300_1000=lambda df: sed_to_flux(df.Flux300_1000, df.LP_Index, 0.3, 1.0),
                 LP_SED1000_3000=lambda df: sed_to_flux(df.Flux1000_3000, df.LP_Index, 1.0, 3.0),
                 LP_SED3000_10000=lambda df: sed_to_flux(df.Flux3000_10000, df.LP_Index, 3.0, 10.0),
                 LP_SED10000_30000=lambda df: sed_to_flux(df.Flux10000_30000, df.LP_Index, 10.0, 30.0),
                 LP_SED30000_100000=lambda df: sed_to_flux(df.Flux30000_100000, df.LP_Index, 30.0, 100.0),
                 PLEC_SED100_300=lambda df: sed_to_flux(df.Flux100_300, df.PLEC_Index, 0.1, 0.3),
                 PLEC_SED300_1000=lambda df: sed_to_flux(df.Flux300_1000, df.PLEC_Index, 0.3, 1.0),
                 PLEC_SED1000_3000=lambda df: sed_to_flux(df.Flux1000_3000, df.PLEC_Index, 1.0, 3.0),
                 PLEC_SED3000_10000=lambda df: sed_to_flux(df.Flux3000_10000, df.PLEC_Index, 3.0, 10.0),
                 PLEC_SED10000_30000=lambda df: sed_to_flux(df.Flux10000_30000, df.PLEC_Index, 10.0, 30.0),
                 PLEC_SED30000_100000=lambda df: sed_to_flux(df.Flux30000_100000, df.PLEC_Index, 30.0, 100.0)
                 )

# byte型=>String型へ

FGL = FGL.astype({'CLASS1': str})
FGL = FGL.astype({'Source_Name': str})
FGL = FGL.astype({'ASSOC1': str})
FGL['CLASS1'].str.decode('utf-8')
FGL.CLASS1 = FGL.CLASS1.replace('(^b)(\')(.*)(  \')', r'\3', regex=True)
FGL.CLASS1 = FGL.CLASS1.replace('(^b)(\')(.*)( \')', r'\3', regex=True)
FGL.CLASS1 = FGL.CLASS1.replace('   ', 'unid', regex=True)
FGL.Source_Name = FGL.Source_Name.replace('(^b)(\')(.*)(\')', r'\3', regex=True)
FGL.Source_Name = FGL.Source_Name.replace('(^3FGL\s)(.*)', r'\2', regex=True)
FGL.ASSOC1 = FGL.ASSOC1.replace('(^b)(\')(.*)(\')', r'\3', regex=True)

FGL = FGL.assign(
    PL_hr12=lambda df: (df.PL_SED300_1000 - df.PL_SED100_300) / (df.PL_SED300_1000 + df.PL_SED100_300),
    PL_hr23=lambda df: (df.PL_SED1000_3000 - df.PL_SED300_1000) / (df.PL_SED1000_3000 + df.PL_SED300_1000),
    PL_hr34=lambda df: (df.PL_SED3000_10000 - df.PL_SED1000_3000) / (df.PL_SED3000_10000 + df.PL_SED1000_3000),
    PL_hr45=lambda df: (df.PL_SED10000_30000 - df.PL_SED3000_10000) / (df.PL_SED10000_30000 + df.PL_SED3000_10000),
    PL_hr56=lambda df: (df.PL_SED30000_100000 - df.PL_SED10000_30000) / (df.PL_SED30000_100000 + df.PL_SED10000_30000),
    LP_hr12=lambda df: (df.LP_SED300_1000 - df.LP_SED100_300) / (df.LP_SED300_1000 + df.LP_SED100_300),
    LP_hr23=lambda df: (df.LP_SED1000_3000 - df.LP_SED300_1000) / (df.LP_SED1000_3000 + df.LP_SED300_1000),
    LP_hr34=lambda df: (df.LP_SED3000_10000 - df.LP_SED1000_3000) / (df.LP_SED3000_10000 + df.LP_SED1000_3000),
    LP_hr45=lambda df: (df.LP_SED10000_30000 - df.LP_SED3000_10000) / (df.LP_SED10000_30000 + df.LP_SED3000_10000),
    LP_hr56=lambda df: (df.LP_SED30000_100000 - df.LP_SED10000_30000) / (df.LP_SED30000_100000 + df.LP_SED10000_30000),
    PLEC_hr12=lambda df: (df.PLEC_SED300_1000 - df.PLEC_SED100_300) / (df.PLEC_SED300_1000 + df.PLEC_SED100_300),
    PLEC_hr23=lambda df: (df.PLEC_SED1000_3000 - df.PLEC_SED300_1000) / (df.PLEC_SED1000_3000 + df.PLEC_SED300_1000),
    PLEC_hr34=lambda df: (df.PLEC_SED3000_10000 - df.PLEC_SED1000_3000) / (
            df.PLEC_SED3000_10000 + df.PLEC_SED1000_3000),
    PLEC_hr45=lambda df: (df.PLEC_SED10000_30000 - df.PLEC_SED3000_10000) / (
            df.PLEC_SED10000_30000 + df.PLEC_SED3000_10000),
    PLEC_hr56=lambda df: (df.PLEC_SED30000_100000 - df.PLEC_SED10000_30000) / (
            df.PLEC_SED30000_100000 + df.PLEC_SED10000_30000)
)

FGL = FGL.drop(['Unc_Flux1000', 'Energy_Flux100',
                'Conf_95_SemiMajor', 'Conf_95_SemiMinor', 'Flux100_300',
                'Conf_68_SemiMajor', 'Conf_68_SemiMinor', 'Conf_68_PosAng', 'Flux1000',
                'Flux100_300', 'Flux300_1000', 'Flux1000_3000',
                'Flux3000_10000', 'Flux10000_30000',
                'LP_SED10000_30000', 'LP_SED1000_3000', 'LP_SED100_300',
                'LP_SED30000_100000', 'LP_SED3000_10000', 'LP_SED300_1000',
                'PLEC_SED10000_30000', 'PLEC_SED1000_3000', 'PLEC_SED100_300',
                'PLEC_SED30000_100000', 'PLEC_SED3000_10000', 'PLEC_SED300_1000',
                'PL_SED10000_30000', 'PL_SED1000_3000', 'PL_SED100_300',
                'PL_SED30000_100000', 'PL_SED3000_10000', 'PL_SED300_1000'], axis=1)
FGL = FGL[FGL['LP_SigCurv'] != 0]
FGL = FGL[FGL['PLEC_SigCurv'] != 0]
FGL = FGL.assign(Variability_Index=lambda df: np.log(df.Variability_Index),
                 Pivot_Energy=lambda df: np.log(df.Pivot_Energy),
                 LP_Flux_Density=lambda df: np.log(df.LP_Flux_Density),
                 PL_Flux_Density=lambda df: np.log(df.PL_Flux_Density),
                 PLEC_Flux_Density=lambda df: np.log(df.PLEC_Flux_Density),
                 LP_Unc_Flux_Density=lambda df: np.log(df.LP_Unc_Flux_Density),
                 PL_Unc_Flux_Density=lambda df: np.log(df.PL_Unc_Flux_Density),
                 PLEC_Unc_Flux_Density=lambda df: np.log(df.PLEC_Unc_Flux_Density),
                 Unc_Energy_Flux100=lambda df: np.log(df.Unc_Energy_Flux100),
                 Flux30000_100000=lambda df: np.log(df.Flux30000_100000),
                 LP_SigCurv=lambda df: np.log(df.LP_SigCurv),
                 PLEC_SigCurv=lambda df: np.log(df.PLEC_SigCurv))

FGL = FGL.drop(['Pivot_Energy',
                'LP_Unc_Flux_Density', 'PL_Unc_Flux_Density', 'PLEC_Unc_Flux_Density',
                'Flux30000_100000'], axis=1)
FGL = FGL.assign(agnness='Non-AGN').assign(agnness=lambda df: df.agnness.mask((df.CLASS1 == 'BCU')
                                                                              | (df.CLASS1 == 'bcu')
                                                                              | (df.CLASS1 == 'BLL')
                                                                              | (df.CLASS1 == 'bll')
                                                                              | (df.CLASS1 == 'FSRQ')
                                                                              | (df.CLASS1 == 'fsrq')
                                                                              | (df.CLASS1 == 'rdg')
                                                                              | (df.CLASS1 == 'RDG')
                                                                              | (df.CLASS1 == 'nlsy1')
                                                                              | (df.CLASS1 == 'NLSY1')
                                                                              | (df.CLASS1 == 'agn')
                                                                              | (df.CLASS1 == 'ssrq')
                                                                              | (df.CLASS1 == 'sey'), 'AGN'))
FGL = FGL.assign(PSRness='Non-PSR').assign(PSRness=lambda df: df.PSRness.mask((df.CLASS1 == 'PSR')
                                                                                          | (df.CLASS1 == 'psr')
                                                                                          , 'PSR'))

FGL_AGN_PSR = FGL[lambda df: (df.PSRness == 'PSR') | (df.agnness == 'AGN')]
# FGL_unid = FGL[lambda df: (df.PSRness == 'Non-PSR') & (df.agnness == 'Non-AGN')]
FGL_unid = FGL[lambda df: (df.CLASS1 == 'unid')]
print('欠損値の処理を行います')
print('AGN&PSR...処理前: ', FGL_AGN_PSR.shape)
print('UNID...処理前: ', FGL_unid.shape)
FGL_AGN_PSR = FGL_AGN_PSR.dropna()
FGL_unid = FGL_unid.dropna()
print('AGN&PSR...処理後: ', FGL_AGN_PSR.shape)
print('UNID...処理後: ', FGL_unid.shape)

FGL_AGN_PSR.to_csv('/Users/kawakami/Desktop/fight/STEP2/STEP2_2/4FGL_after_drop.csv')

FGL_AGN_PSR = FGL_AGN_PSR.drop(['CLASS1', 'ASSOC1', 'Source_Name', 'Conf_95_PosAng', 'agnness'], axis=1)
FGL_unid = FGL_unid.drop(['CLASS1', 'ASSOC1', 'Source_Name', 'Conf_95_PosAng', 'agnness'], axis=1)

print('PSR :', FGL_AGN_PSR[FGL_AGN_PSR.PSRness == 'PSR'].shape[0])
print('AGN :', FGL_AGN_PSR[FGL_AGN_PSR.PSRness == 'Non-PSR'].shape[0])
print('PSR + AGN :', FGL_AGN_PSR.shape[0])
print('Unid :', FGL_unid.shape[0])

FGL_AGN_PSR_data = FGL_AGN_PSR.select_dtypes(exclude='object')
# 正規化
FGL_AGN_PSR_data_processed = FGL_AGN_PSR_data.iloc[:, :].apply(stats.zscore, axis=0)
FGL_AGN_PSR_data_processed['PSRness'] = FGL_AGN_PSR.PSRness
FGL_AGN_PSR_data_processed.to_csv('/Users/kawakami/Desktop/fight/STEP2/STEP2_3/FGLstd.csv')

FGL_unid_data = FGL_unid.select_dtypes(exclude='object')
FGL_unid_data.to_csv('/Users/kawakami/Desktop/fight/STEP2/STEP2_3/FGL_unid_プロット用.csv')
FGL_unid_data_processed = FGL_unid_data.iloc[:, :].apply(stats.zscore, axis=0)
FGL_unid_data_processed['PSRness'] = FGL_unid.PSRness
FGL_unid_data_processed.drop(['PSRness'], axis=1)
FGL_unid_data_processed.to_csv('/Users/kawakami/Desktop/fight/STEP2/STEP2_3/FGL_unid_解析用.csv')

