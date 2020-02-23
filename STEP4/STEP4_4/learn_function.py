# coding=utf-8

# DataFrameを使う
import pandas as pd
# 自分で設定した指標を学習精度の評価に適用する
from sklearn.metrics import make_scorer
# K-分割交差検証とパラメータフィッティング
from sklearn.model_selection import GridSearchCV, StratifiedKFold
# Feature Warningを無視する
from warnings import simplefilter
# DataFrameのデータの統計を確認する
import pandas_profiling as ppf

# 学習モデル

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# GridSearch用のパラメタ探索範囲と評価指標に関する自作パッケージ
from function.param import *
from function.evaluation_function import *
import numpy as np
# Feature Warningを無視する
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)
# シード値を設定しておく
np.random.seed(31415)

# 最大表示行数の指定
pd.set_option('display.max_rows', None)


def fgl(csv_path):
    """
    欠損値・標準化処理済みのcsvデータを読み込みDataFrameに変換します.
    :return: 解析データ(DataFrame)
    """
    FGL = pd.read_csv(csv_path, header=0).drop(["Unnamed: 0"], axis=1)
    # 読み込んだデータに対する統計プロファイルをHTMLに出力

    return FGL


def statistic_info(FGL, path):
    """
    指定したDataFrame型のデータの特徴量同士の相関や欠損値など統計的な情報をHTMLで出力します.
    :param FGL: DataFrame
    :return: なし (統計データプロファイルを作成)
    """
    path = path + "/ana.html"
    ppf.ProfileReport(FGL).to_file(outputfile=path)


def save_train_set(FGL):
    """
    指定したDataFrame型のデータを特徴量と出力に分けます
    :param FGL: DataFrame
    :return: 特徴量データ(DataFrame), 結果データ(DataFrame)
    """
    X = FGL.drop("FSRQness", axis=1)
    y = FGL.loc[:, 'FSRQness']
    return X, y


def divided_trainset(FGL):
    """
    DataFrameを訓練データとテストデータに分け、指定したデータ分割数に応じたデータセットが入ったリストを返します。
    :param FGL:
    :param i: データの分割数(3)
    :return: 訓練データのリスト、テストデータのリスト(各要素はDataFrame)
    """
    # DataFrame
    X, y = save_train_set(FGL)

    # 層化抽出法を用いたK-分割交差検証
    skf = StratifiedKFold(n_splits=3, random_state=1)
    skf.get_n_splits(X, y)

    X_train_list, y_train_list, X_test_list, y_test_list = [], [], [], []

    for train_index, test_index in StratifiedKFold(n_splits=3).split(X, y):
        X_train_list.append(X.iloc[train_index])
        y_train_list.append(y.iloc[train_index])
        X_test_list.append(X.iloc[test_index])
        y_test_list.append(y.iloc[test_index])

    return X_train_list, y_train_list, X_test_list, y_test_list


def choose_trainset(FGL, i):
    """
    データリストの中から対応するの訓練データとテストデータのデータセットを返します。
    :param i:
    :return: 訓練データ(np.array)とテストデータ(np.array)
    """
    X_train_list, y_train_list, X_test_list, y_test_list = divided_trainset(FGL)

    X_train = X_train_list[i].values
    y_train = pd.get_dummies(y_train_list[i].values)['FSRQ']
    X_test = X_test_list[i].values
    y_test = pd.get_dummies(y_test_list[i].values)['FSRQ']

    return X_train, y_train, X_test, y_test


def train_data_info(FGL, i):
    """
    訓練データ、テストデータの大きさをcsvファイルに記録します.
    :return: なし
    """
    X_train_list, y_train_list, X_test_list, y_test_list = divided_trainset(FGL)
    X_train = X_train_list[i].values
    X_test = X_test_list[i].values
    # csvにデータの大きさの情報を保存
    path = "/Users/kawakami/Desktop/fight/STEP4/STEP4_4/result/DataSize{}.csv".format(i)
    pd.DataFrame(data={'result name': [X_train.shape + X_test.shape, X_train.shape, X_test.shape],
                       'result shape': ['X', 'X_train', 'X_test']},
                 columns=['result shape', 'result name']).to_csv(path_or_buf=path)


def learn(FGL, i):
    """
    各モデルに指定した訓練データ、テストデータでモデルの分類精度を記録します.
    :param i:
    :return:
    """
    X_train, y_train, X_test, y_test = choose_trainset(FGL, i-1)

    # Tssを学習精度の評価に適用する
    TssScore = make_scorer(tss, greater_is_better=True)
    # 学習結果を保存する配列(TSS、モデル、最適パラメタ、confusionMatrix)
    tss_FGL = []
    model = []
    params = []
    cm_score = []

    svm_poly_model(TssScore, X_test, X_train, cm_score, model, params, tss_FGL, y_test, y_train)

    svm_rbf_model(TssScore, X_test, X_train, cm_score, model, params, tss_FGL, y_test, y_train)

    svm_linear_model(TssScore, X_test, X_train, cm_score, model, params, tss_FGL, y_test, y_train)

    gaussian_nb_model(X_test, X_train, cm_score, model, params, tss_FGL, y_test, y_train)

    bernoulli_nb_model(X_test, X_train, cm_score, model, params, tss_FGL, y_test, y_train)

    logistic_reg_model(X_test, X_train, cm_score, model, params, tss_FGL, y_test, y_train)

    knn_model(TssScore, X_test, X_train, cm_score, model, params, tss_FGL, y_test, y_train)

    gb_model(TssScore, X_test, X_train, cm_score, model, params, tss_FGL, y_test, y_train)

    random_forest_model(X_test, X_train, cm_score, model, params, tss_FGL, y_test, y_train)

    CM = np.array(cm_score).T

    # csvに予測精度結果を保存
    path = "/Users/kawakami/Desktop/fight/STEP4/STEP4_4/result/model_acc_dataset{}.csv".format(i)
    pd.DataFrame(data={'model name': model, 'TSS Value': tss_FGL, 'setting': params, 'TP': CM[3], 'FN': CM[2], 'TN': CM[0], 'FP': CM[1]},
                 columns=['model name', 'TSS Value', 'setting', 'TP', 'FN', 'TN', 'FP']).to_csv(
        path_or_buf=path, encoding='shift_jis')


def random_forest_model(X_test, X_train, cm_score, model, params, tss_FGL, y_test, y_train):
    # ランダムフォレスト
    random_forest = RandomForestClassifier()
    random_forest.fit(X_train, y_train)
    model.append("ランダムフォレスト")
    tss_FGL.append(tss(y_test, random_forest.predict(X_test)))
    params.append("no-setting")
    cm_score.append(class_result(y_test, random_forest.predict(X_test)))


def svm_poly_model(TssScore, X_test, X_train, cm_score, model, params, tss_FGL, y_test, y_train):
    print("サポートベクターマシン(ploy)")
    poly = SVC()
    SVM_poly = GridSearchCV(poly, param_grid=param_poly('poly'), cv=3, scoring=TssScore, verbose=1,
                            return_train_score=False, n_jobs=-1)
    SVM_poly.fit(X_train, y_train)
    model.append("サポートベクターマシン(ploy)")
    tss_FGL.append(tss(y_test, SVM_poly.best_estimator_.predict(X_test)))
    params.append(SVM_poly.best_params_)
    cm_score.append(class_result(y_test, SVM_poly.best_estimator_.predict(X_test)))


def svm_linear_model(TssScore, X_test, X_train, cm_score, model, params, tss_FGL, y_test, y_train):
    print("サポートベクターマシン(linear)")
    linear = SVC()
    SVM_linear = GridSearchCV(linear, param_grid=param_linear('linear'), cv=3, scoring=TssScore, verbose=1,
                              return_train_score=False, n_jobs=-1)
    SVM_linear.fit(X_train, y_train)
    model.append("サポートベクターマシン(linear)")
    tss_FGL.append(tss(y_test, SVM_linear.best_estimator_.predict(X_test)))
    params.append(SVM_linear.best_params_)
    cm_score.append(class_result(y_test, SVM_linear.best_estimator_.predict(X_test)))


def svm_rbf_model(TssScore, X_test, X_train, cm_score, model, params, tss_FGL, y_test, y_train):
    print("サポートベクターマシン(RBF)")
    rbf = SVC()
    SVM_rbf = GridSearchCV(rbf, param_grid=param_rbf('rbf'), cv=3, scoring=TssScore, verbose=1,
                           return_train_score=False, n_jobs=-1)
    SVM_rbf.fit(X_train, y_train)
    model.append("サポートベクターマシン(RBFカーネル)")
    tss_FGL.append(tss(y_test, SVM_rbf.best_estimator_.predict(X_test)))
    params.append(SVM_rbf.best_params_)
    cm_score.append(class_result(y_test, SVM_rbf.best_estimator_.predict(X_test)))


def gb_model(TssScore, X_test, X_train, cm_score, model, params, tss_FGL, y_test, y_train):
    print("勾配ブースティング")
    grb = GridSearchCV(GradientBoostingClassifier(), param_grid=param_gbr(), cv=3, scoring=TssScore, verbose=1,
                       return_train_score=False, n_jobs=-1)
    grb.fit(X_train, y_train)
    model.append('勾配ブースティング')
    tss_FGL.append(tss(y_test, grb.best_estimator_.predict(X_test)))
    params.append(grb.best_params_)
    cm_score.append(class_result(y_test, grb.best_estimator_.predict(X_test)))


def knn_model(TssScore, X_test, X_train, cm_score, model, params, tss_FGL, y_test, y_train):
    print("k近傍法")
    kNC = GridSearchCV(KNeighborsClassifier(), param_grid=param_knn(), cv=3, scoring=TssScore, verbose=1,
                       return_train_score=False, n_jobs=-1)
    kNC.fit(X_train, y_train)
    model.append("k近傍法")
    tss_FGL.append(tss(y_test, kNC.best_estimator_.predict(X_test)))
    params.append(kNC.best_params_)
    cm_score.append(class_result(y_test, kNC.best_estimator_.predict(X_test)))


def logistic_reg_model(X_test, X_train, cm_score, model, params, tss_FGL, y_test, y_train):
    print("ロジスティック回帰")
    LR = LogisticRegression(fit_intercept=True, class_weight='balanced', n_jobs=-1)
    LR.fit(X_train, y_train)
    model.append("ロジスティック回帰")
    tss_FGL.append(tss(y_test, LR.predict(X_test)))
    params.append("no-setting")
    cm_score.append(class_result(y_test, LR.predict(X_test)))


def bernoulli_nb_model(X_test, X_train, cm_score, model, params, tss_FGL, y_test, y_train):
    print("ベルヌーイナイーブベイズ")
    bernoulli_nb = BernoulliNB()
    bernoulli_nb.fit(X_train, y_train)
    model.append("ベルヌーイナイーブベイズ")
    tss_FGL.append(tss(y_test, bernoulli_nb.predict(X_test)))
    params.append("no-setting")
    cm_score.append(class_result(y_test, bernoulli_nb.predict(X_test)))


def gaussian_nb_model(X_test, X_train, cm_score, model, params, tss_FGL, y_test, y_train):
    print("ガウシアンナイーブベイズ")
    gaussian_nb = GaussianNB()
    gaussian_nb.fit(X_train, y_train)
    model.append("ガウシアンナイーブベイズ")
    tss_FGL.append(tss(y_test, gaussian_nb.predict(X_test)))
    params.append("no-setting")
    cm_score.append(class_result(y_test, gaussian_nb.predict(X_test)))


def corr_column(df, threshold):
    df_corr = df.corr()
    df_corr = abs(df_corr)
    columns = df_corr.columns

    # 対角線の値を0にする
    for i in range(0, len(columns)):
        df_corr.iloc[i, i] = 0

    while True:
        columns = df_corr.columns
        max_corr = 0.0
        query_column = None
        target_column = None

        df_max_column_value = df_corr.max()
        max_corr = df_max_column_value.max()
        query_column = df_max_column_value.idxmax()
        target_column = df_corr[query_column].idxmax()

        if max_corr < threshold:
            # しきい値を超えるものがなかったため終了
            break
        else:
            # しきい値を超えるものがあった場合
            delete_column = None
            saved_column = None

            # その他との相関の絶対値が大きい方を除去
            if sum(df_corr[query_column]) <= sum(df_corr[target_column]):
                delete_column = target_column
                saved_column = query_column
            else:
                delete_column = query_column
                saved_column = target_column

            # 除去すべき特徴を相関行列から消す（行、列）
            df_corr.drop([delete_column], axis=0, inplace=True)
            df_corr.drop([delete_column], axis=1, inplace=True)

    return df_corr.columns


