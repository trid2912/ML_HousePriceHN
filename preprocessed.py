import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.model_selection import train_test_split
import sys


def IQR(df, column, qi = 1.5):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    df_processed = df[df[column] > (Q1 - qi*IQR)]
    df_processed = df_processed[df_processed[column] < (Q3 + qi*IQR)]
    return df_processed

def data_preprocess(directory, train_test=True):
    df = pd.read_csv(directory)
    df = df.rename(columns = {"Ngày": "Date", "Địa chỉ" :"Address", "Quận": "District", "Huyện": "Ward", "Loại hình nhà ở": "House_type",
                     "Giấy tờ pháp lý": "Legal_status","Số tầng": "num_floors","Số phòng ngủ": "num_bedrooms", "Diện tích": "Area",
                     "Dài": "Length", "Rộng": "Width","Giá/m2": "Price"})
    df.drop(df.columns[0], axis =1, inplace = True)
    df["Legal_status"].fillna("Không có sổ", inplace=True)
    df.dropna(inplace=True)
    df.drop(["Date", "Length", "Width", "Address"], axis=1, inplace=True)
    df['Area'] = df['Area'].str.replace(' m²','').str.strip().astype(float)
    df.loc[df['Price'].str.contains(' tỷ/m²', na = False), 'Price'] = df.loc[df['Price'].str.contains(' tỷ/m²', na = False), 'Price'].str.replace(' tỷ/m²','').str.replace('.','').str.replace(',','.').astype(float) * 1e3
    df.loc[df['Price'].str.contains(' triệu/m²', na = False), 'Price'] = df.loc[df['Price'].str.contains(' triệu/m²', na=False), 'Price'].str.replace(' triệu/m²','').str.replace(',','.').astype(float)
    df.loc[df['Price'].str.contains(' đ/m²', na = False), 'Price'] = df.loc[df['Price'].str.contains(' đ/m²', na = False), 'Price'].str.replace(' đ/m²','').str.replace('.','').astype(float) * 1e-6
    df = df[df['num_floors'] != 'Nhiều hơn 10']
    df = df[df['num_bedrooms'] != 'nhiều hơn 10 phòng']
    df.loc[df['num_bedrooms'].str.contains(' phòng', na=False), 'num_bedrooms'] = df.loc[df['num_bedrooms'].str.contains(' phòng', na=False), 'num_bedrooms'].str.replace(' phòng','').str.replace(',','.').astype(float)
    df["num_floors"] = df["num_floors"].astype(float)
    df = df[df["District"].str.slice(0,4) == "Quận"]
    df.drop(["District"], axis=1, inplace = True)
    df = df[((df["Area"] >= 30) & (df["Area"] <= 300))]
    df = df[df["num_floors"] <= 8]
    dfa = IQR(df, 'Price')
    df = dfa.copy()
    df.loc[df["Legal_status"] == "Đã có sổ", "Legal_status"] = 4
    df.loc[df["Legal_status"] == "Đang chờ sổ", "Legal_status"] = 3
    df.loc[df["Legal_status"] == "Giấy tờ khác", "Legal_status"] = 2
    df.loc[df["Legal_status"] == "Không có sổ", "Legal_status"] = 1
    df.loc[df["House_type"] == "Nhà biệt thự", "House_type"] = 4
    df.loc[df["House_type"] == "Nhà mặt phố, mặt tiền", "House_type"] = 2
    df.loc[df["House_type"] == "Nhà phố liền kề", "House_type"] = 3
    df.loc[df["House_type"] == "Nhà ngõ, hẻm", "House_type"] = 1
    encoder = ce.BinaryEncoder(cols=['Ward'], return_df=True)
    df = encoder.fit_transform(df)
    X = df.drop(['Price'], axis=1)

    y = df[['Price']]

    X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(), y.to_numpy(), test_size=0.2, random_state=4)
    if train_test:
        return X_train, X_test, y_train, y_test
    else:
        print(dfa.loc[34, :])
        return X.to_numpy(), y.to_numpy(), 34
        

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = data_preprocess(sys.argv[1])
    print(X_train.shape[0] + X_test.shape[0])
