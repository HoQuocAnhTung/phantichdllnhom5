import pandas as pd
import os
import matplotlib.pyplot as plt
import xlwings as xw
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
import numpy as np

folder_path = r"C:\DataTung\databtlphantichdl\data"

fdata_total = os.path.join(folder_path,"all.csv")

list_file = os.listdir(folder_path)

#create app
app = xw.App(add_book = False)
list_total_wb = []
b = 0
for file in list_file:
    if file.split(".")[-1] == "csv":
        fpath = os.path.join(folder_path,file)
        wb = app.books.open(fpath)
        sht = wb.sheets[0]
        lr = sht.range("A" + str(sht.cells.last_cell.row)).end("up").row
        lc = sht.range("A2").end("right").column
        if b == 0:
            data_sht = sht.range((1,1),(lr,lc)).value
            b = 1
        else:
            data_sht = sht.range((2, 1),(lr, lc)).value
        list_total_wb.extend(data_sht)
        wb.close()

wb_total = app.books.add()
sht_total = wb_total.sheets[0]
sht_total.range("A1").value = list_total_wb
sht_total.range("1:1").apt.Font.Bold = True
wb_total.save(fdata_total)

data = pd.read_csv("all.csv")
print(data)
data['Order Date'] = pd.to_datetime(data['Order Date'], format='%m/%d/%Y', errors='coerce')  # Thay đổi định dạng thời gian
data.dropna(subset=['Order Date'], inplace=True)  # Xóa các hàng có dữ liệu không hợp lệ
        # 7. Trích xuất cột `Month` từ cột `Order Date`
data['Month'] = data['Order Date'].dt.month
        # 8. Tính tổng doanh thu cho mỗi tháng
data.head()
data1 = data.dropna(how='all')
data2 = data1[data1['Month'] != 'Or']
data2.head()

data2.to_csv('all.csv')

dd = df.shape
print(dd)

import pandas as pd

df = pd.read_csv('mtcars.csv')
#  Chọn các cột dữ liệu số (numeric columns)
df_numeric = df.select_dtypes(include=['number'])
print(df_numeric)
#Bởi muốn tính TBC, ta phải chuyển dữ liệu về dạng số

# đếm các dữ liệu không bị khuyết
data_count = df_numeric.count()
print(data_count)
# Tính và in ra trung bình cộng theo hàng (axis=1)
row_means = df_numeric.mean(axis=1)
print("Trung bình cộng theo hàng:")
print(row_means)

# Tính và in ra trung bình cộng theo cột (axis=0)
column_means = df_numeric.mean(axis=0)
print("Trung bình cộng theo cột:")
print(column_means)

# Tính median của từng cột
column_medians = df_numeric.median()

print("Median của từng cột:")
print(column_medians)

# Tính mode của từng cột
column_modes = df_numeric.mode()

print("Mode của từng cột:")
print(column_modes)

# Tính giá trị max của từng cột
column_max = df_numeric.max()
print("Giá trị max của từng cột:")
print(column_max)

# Tính giá trị min của từng cột
column_min = df_numeric.min()
print("\nGiá trị min của từng cột:")
print(column_min)

# Tính Q1, Q2 , Q3 cho từng cột
column_q1 = df_numeric.quantile(0.25)
column_q2 = df_numeric.median()
column_q3 = df_numeric.quantile(0.75)
column_IQR = column_q3 - column_q1
print("Q1 của từng cột:")
print(column_q1)

print("\nMedian của từng cột:")
print(column_q2)

print("\nQ3 của từng cột:")
print(column_q3)

print("\nIQR của từng cột:")
print(column_IQR)

# Tính phương sai của từng cột
column_variances = df_numeric.var()
print("Phương sai của từng cột:")
print(column_variances)

# Tính độ lệch chuẩn của từng cột
column_std_devs = df_numeric.std()
print("\nĐộ lệch chuẩn của từng cột:")
print(column_std_devs)


# Tạo bảng thống kê (tự tạo bảng để giống với bảng đề bài yêu cầu)
def descriptive(data_count,column_min,column_max,column_medians,column_modes,column_q1,column_q2,column_q3,column_IQR,column_variances,column_std_devs):
        data = {'Count': [i for i in data_count ],
                'min': [i for i in column_min ],
                'max': [i for i in column_max ],
                'median': [i for i in column_medians ],
                'mode': [i for i in column_modes.values[0]],
                'Q1': [i for i in column_q1 ],
                'Q2': [i for i in column_q2 ],
                'Q3': [i for i in column_q3 ],
                'IQR': [i for i in column_IQR ],
                'Variance': [i for i in column_variances ],
                'stdev': [i for i in column_std_devs ],
                } # dữ liệu đang ở dạng dic
        df1 = pd.DataFrame(data) # convert về dạng pandas
        df1.index=df_numeric.keys() # keys sẽ trả về tên của các cột( features)
        data_complete = df1.transpose() # transpose để chuyển hàng về cột, cột về hàng
# Thêm một cột mới vào đầu DataFrame
        new_column_data = ['count','min','max','median','mode','Q1','Q2','Q3','IQR','Variance','stdev']
        column_name = ' '
        data_complete.insert(loc=0, column=column_name, value=new_column_data)
        print(data_complete)
        data_complete.to_csv('Thong_ke_1.txt', sep='\t', index=False)

descriptive(data_count,column_min,column_max,column_medians,column_modes,column_q1,column_q2,column_q3,column_IQR,column_variances,column_std_devs)
print('---------------------------------------------------------------------------------------------------------------------------------------------')


tt = df.isna().sum()
print(tt)

df['Order ID'].fillna(df['Order ID'].mode()[0], inplace=True)
df['Product'].fillna(df['Product'].mode()[0], inplace=True)
df['Quantity Ordered'].fillna(df['Quantity Ordered'].mean(), inplace=True)
df['Price Each'].fillna(df['Price Each'].mean(), inplace=True)
df['Order Date'].fillna(df['Order Date'].mean(), inplace=True)
df['Purchase'].fillna(df['Purchase'].mean(), inplace=True)


float_cols = ['Order ID', 'Product', 'Quantity Ordered', 'Price Each',
       'Order Date', 'Purchase']
df_mean = df_combined[float_cols].mean().values
df_std = df_combined[float_cols].std().values
print(df_mean, df_std)
with open('mean_std.txt', "wb") as f:
  f.write(df_mean)
  f.write(df_std)

scaler = StandardScaler()
for col in float_cols:
  df_combined[col] = scaler.fit_transform(df_combined[[col]])