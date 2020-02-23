# coding=utf-8

from STEP3.STEP3_4.learn_function import *

FGL = fgl("/Users/kawakami/Desktop/fight/STEP3/STEP3_2/4FGL_AGN_PSR_std.csv").loc[:,
      ['Conf_68_SemiMajor',
       'Conf_95_PosAng', 'Signif_Avg', 'Pivot_Energy',
       'Unc_Energy_Flux100', 'PL_Flux_Density', 'Unc_PL_Flux_Density',
       'PL_Index', 'Unc_PL_Index', 'LP_Index', 'Unc_LP_Index', 'LP_beta',
       'PLEC_Index', 'Unc_PLEC_Index', 'PLEC_Expfactor', 'Unc_PLEC_Expfactor',
       'PLEC_Exp_Index', 'PLEC_SigCurv', 'Energy_Flux100', 'Variability_Index',
       'Frac_Variability', 'Unc_Frac_Variability', 'Flux50_100',
       'Flux1000_3000', 'Flux10000_30000', 'Flux30000_100000', 'PSRness']]

ppf.ProfileReport(FGL).to_file(outputfile="/Users/kawakami/Desktop/fight/STEP3/STEP3_4/result")

for i in range(len(FGL.columns)):
    if not FGL.columns[i] in corr_column(FGL, 0.9):
        print(FGL.columns[i], ",", end="")

X, y = save_train_set(FGL)

X_train, y_train, X_test, y_test = choose_trainset(FGL, 1)
print("X", X.shape, "X_train: ", X_train.shape, " X_test: ", X_test.shape)
learn(FGL, 1)
learn(FGL, 2)
learn(FGL, 3)
