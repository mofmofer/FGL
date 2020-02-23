# coding=utf-8

from STEP1.STEP1_4.learn_function import *

FGL = fgl("/Users/kawakami/Desktop/fight/STEP1/STEP1_2/4FGL_AGN_PSR_std.csv").loc[:,
      ['RAJ2000', 'DEJ2000', 'GLON', 'GLAT', 'Conf_68_SemiMajor',
       'Conf_95_PosAng', 'Signif_Avg', 'Pivot_Energy',
       'Unc_Energy_Flux100', 'PL_Flux_Density', 'Unc_PL_Flux_Density',
       'PL_Index', 'Unc_PL_Index', 'LP_Index', 'Unc_LP_Index', 'LP_beta',
       'PLEC_Index', 'Unc_PLEC_Index', 'PLEC_Expfactor', 'Unc_PLEC_Expfactor',
       'PLEC_Exp_Index', 'PLEC_SigCurv', 'Energy_Flux100', 'Variability_Index',
       'Frac_Variability', 'Unc_Frac_Variability', 'Flux50_100',
       'Flux1000_3000', 'Flux10000_30000', 'Flux30000_100000', 'PSRness']]

for i in range(len(FGL.columns)):
    if not FGL.columns[i] in corr_column(FGL, 0.9):
        print(FGL.columns[i], ",", end="")

X, y = save_train_set(FGL)

X_train, y_train, X_test, y_test = choose_trainset(FGL, 1)
print("X", X.shape, "X_train: ", X_train.shape, " X_test: ", X_test.shape)

# Tssを学習精度の評価に適用する
TssScore = make_scorer(tss, greater_is_better=True)

# 学習結果を保存する配列(TSS、モデル、最適パラメタ、confusionMatrix)
tss_FGL = []
model = []
params = []
cm_score = []
print("勾配ブースティング")
grb = GridSearchCV(GradientBoostingClassifier(), param_grid=param_gbr(), cv=3, scoring=TssScore, verbose=1,
                   return_train_score=False, n_jobs=-1)
grb.fit(X_train, y_train)

model.append('勾配ブースティング')
clf = grb.best_estimator_

tss_FGL.append(tss(y_test, clf.predict(X_test)))
params.append(grb.best_params_)
cm_score.append(class_result(y_test, grb.best_estimator_.predict(X_test)))

feature_importance = clf.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())
print(FGL.drop('PSRness', axis =1))
label = FGL.drop('PSRness', axis =1).columns
print(label)
print(feature_importance)
plt.xlabel('feature importance')
plt.barh(label, feature_importance, tick_label=label, align="center")
plt.savefig('/Users/kawakami/Desktop/fight/STEP1/STEP1_4/result/figure.png')
