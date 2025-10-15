import pandas as pd
from dataSplit import DataSplitter

# Đọc dữ liệu đã tiền xử lý
df_nostopwords = pd.read_csv("../dataset/data_preprocessed_no-stopwords.csv")
df_stopwords = pd.read_csv("../dataset/data_preprocessed_stopwords.csv")

data = DataSplitter(df_nostopwords, text_col="content", label_col="label")
# data = DataSplitter(df_stopwords, text_col="content", label_col="label")

# Hold-out
# print("Hold-out:", data.holdout_X_train[:10])
# print("\n ------------------------------- \n")
# print("Hold-out:", data.holdout_y_train[:10])
# print("\n ------------------------------- \n")
# print("Hold-out:", data.holdout_X_test[:10])
# print("\n ------------------------------- \n")
# print("Hold-out:", data.holdout_y_test[:10])

# K-Fold
# print("K-Fold:", data.kfold_X_train, data.kfold_y_train, data.kfold_X_test, data.kfold_y_test)


# # Stratified
# print("Stratified:", data.strat_X_train, data.strat_y_train, data.strat_X_test, data.strat_y_test)