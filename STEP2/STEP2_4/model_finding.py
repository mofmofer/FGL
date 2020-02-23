# coding=utf-8

from STEP2.STEP2_4.learn_function import *

FGL = fgl("/Users/kawakami/Desktop/fight/STEP2/STEP2_3/FGLstd.csv")

print(FGL.shape)

ppf.ProfileReport(FGL).to_file(outputfile="/Users/kawakami/Desktop/fight/STEP2/STEP2_4/result/ana.html")

for i in range(len(FGL.columns)):
    if not FGL.columns[i] in corr_column(FGL, 0.9):
        print(FGL.columns[i], ",", end="")

X, y = save_train_set(FGL)

X_train, y_train, X_test, y_test = choose_trainset(FGL, 1)
print("X", X.shape, "X_train: ", X_train.shape, " X_test: ", X_test.shape)
learn(FGL, 1)
learn(FGL, 2)
learn(FGL, 3)
