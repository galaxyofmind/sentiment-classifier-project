# Thư viện
import pandas as pd 
from preprocessNLP import Preprocessor


#  Đọc dữ liệu cho việc huấn luyện mô hình
df=pd.read_csv("../dataset/new_data.csv")

print("Số dòng dữ liệu:", len(df))
print(df.head())
print(df['label'].value_counts())

print("\n ------------------------------- \n")

# Đổi các giá trị nhãn "NEU" thành "POS", 1/0
# df['label'] = df['label'].replace('NEU', 'POS')

# Encoding label: NEG -> 0, POS -> 1
# Xóa các dòng có nhãn là NEU
df = df[df['label'] != 'NEU'].reset_index(drop=True)
df['label'] = df['label'].replace('NEG', 0)
df['label'] = df['label'].replace('POS', 1)
df['label'] = df['label'].astype(int)
print(df['label'].value_counts())

print("\n ------------------------------- \n")

# Chuyển đổi cột 'content' sang kiểu string
df['content'] = df['content'].astype(str)
print(df.dtypes)

print("\n ------------------------------- \n")

# Bỏ dữ liệu không cần thiết
df = df.iloc[:, :-1]
df = df.dropna()
print(df.dtypes) 


# Xử lý văn bản (có loại bỏ stopwords)
temp = df.copy()
preprocess_text = Preprocessor(stopwords_file="../dataset/vietnamese-stopwords.txt", remove_stopwords=True, lemmatization=True)
for i in range(len(df)):
    temp.loc[i, "content"] = preprocess_text.preprocess_text(df['content'][i])

    
# lưu dữ liệu đã tiền xử lý vào file csv
temp.to_csv("../dataset/data_preprocessed_no-stopwords.csv", index=False, header=True)
print("Lưu thành công file data_preprocessed_no-stopwords.csv")

# Xử lý văn bản 
temp = df.copy()
preprocess_text = Preprocessor(stopwords_file="../dataset/vietnamese-stopwords.txt", remove_stopwords=False, lemmatization=True)
for i in range(len(df)):
    temp.loc[i, "content"] = preprocess_text.preprocess_text(df['content'][i])
    
# lưu dữ liệu đã tiền xử lý vào file csv
temp.to_csv("../dataset/data_preprocessed_stopwords.csv", index=False, header=True)
print("Lưu thành công file data_preprocessed_stopwords.csv")
