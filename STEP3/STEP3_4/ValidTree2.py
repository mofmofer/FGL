# coding=utf-8
import matplotlib.pyplot as plt

from STEP3.STEP3_4.learn_function import *

FGL = fgl("/Users/kawakami/Desktop/fight/STEP3/STEP3_2/4FGL_AGN_PSR_std.csv").loc[:,
      ['ROI_num', 'Signif_Avg',
       'Pivot_Energy', 'Unc_Flux1000', 'PL_Flux_Density',
       'Unc_PL_Flux_Density', 'PL_Index', 'Unc_PL_Index', 'LP_Index',
       'Unc_LP_Index', 'LP_beta', 'LP_SigCurv', 'PLEC_Index', 'Unc_PLEC_Index',
       'PLEC_Expfactor', 'Unc_PLEC_Expfactor', 'PLEC_Exp_Index', 'Npred',
       'Variability_Index', 'Frac_Variability', 'Unc_Frac_Variability',
       'Flux1000_3000', 'Flux30000_100000', 'PSRness']]

ppf.ProfileReport(FGL).to_file(outputfile="/Users/kawakami/Desktop/fight/STEP3/STEP3_4/modified_4FGL_AGN_PSR_std.html")

print("相関の高い特徴量を表示します")
for i in range(len(FGL.columns)):
    if not FGL.columns[i] in corr_column(FGL, 0.9):
        print(FGL.columns[i], ",", end="")


importance = []
for i in range(0,3):
    X, y = save_train_set(FGL)
    X_train, y_train, X_test, y_test = choose_trainset(FGL, i)
    print("データの形を表示します", "X", X.shape, "X_train: ", X_train.shape, " X_test: ", X_test.shape)

    # Tssを学習精度の評価に適用する
    TssScore = make_scorer(tss, greater_is_better=True)

    grb = GridSearchCV(GradientBoostingClassifier(), param_grid=param_gbr(), cv=3, scoring=TssScore, verbose=1,
                   return_train_score=False, n_jobs=-1)

    grb.fit(X_train, y_train)
    clf = grb.best_estimator_
    print(grb.best_params_)
    print(tss(y_test, clf.predict(X_test)))
    print(class_result(y_test, grb.best_estimator_.predict(X_test)))

    feature_importance = clf.feature_importances_
    # feature_importance = 100.0 * (feature_importance / feature_importance.max())
    feature_importance = feature_importance * (100 / feature_importance.sum())
    importance.append(feature_importance)
FGL = FGL.drop('PSRness', axis=1)

label = FGL.columns
feature_importance = (importance[0] + importance[1] +importance[2])/3
df = pd.DataFrame({
    'columns': FGL.columns,
    'rank': feature_importance,
})

df = df.sort_values('rank')
df.to_csv("/Users/kawakami/Desktop/fight/STEP3/STEP3_4/result/feature2.csv")
plt.figure(figsize=(15, 10))

plt.xlabel('feature importance', fontsize=18)
# plt.barh(label, feature_importance, tick_label=label, align="center")
plt.barh(df['columns'].values, df['rank'].values, tick_label=df['columns'].values, align="center")
plt.savefig('/Users/kawakami/Desktop/fight/STEP3/STEP3_4/result/figure2.png', format='png', dpi=100)
