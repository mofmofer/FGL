"""
精度の評価、結果を出力するような関数です.
"""


# CMパラメータ
def con_matrix(y_test, pre):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, pre)
    TN, FP, FN, TP = cm.flatten()
    return float(TN), float(FP), float(FN), float(TP)


# CMパラメータ をPrint
def class_result(y_test, pre):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, pre)
    TN, FP, FN, TP = cm.flatten()
    return [TN, FP, FN, TP]


# TSSスコア(AGNの正解率 - PSRの不正解率)
def tss(y_test, pre):
    TN, FP, FN, TP = con_matrix(y_test, pre)
    TSS = TP / (TP + FN) - FP / (FP + TN)
    return TSS
