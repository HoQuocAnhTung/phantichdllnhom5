import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Đọc dữ liệu từ file "tongSale.csv"
df = pd.read_csv("tongSale.csv")

# Hiển thị 5 dòng đầu tiên
print(df.head().to_markdown(index=False, numalign="left", stralign="left"))

# Hiển thị thông tin về các cột và kiểu dữ liệu
print(df.info())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Tạo X và y
X = df[['Month']]
y = df['Sales']

# Chia dữ liệu thành tập huấn luyện (train) và tập kiểm tra (test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hiển thị 5 dòng đầu tiên
print(X_train.head().to_markdown(index=False, numalign="left", stralign="left"))

# Hiển thị thông tin về các cột và kiểu dữ liệu
print(X_train.info())

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Huấn luyện mô hình
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Dự đoán trên tập train và tính MSE, R2 thẩm định
y_train_pred = model.predict(X_train_scaled)
mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

# Dự đoán trên tập test và tính MSE, R2 thực tiễn
y_test_pred = model.predict(X_test_scaled)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

# Hiển thị kết quả
print("------Kết quả đánh giá-------")
print("MSE (thẩm định):", mse_train)
print("R-squared (thẩm định):", r2_train)
print("MSE (thực tiễn):", mse_test)
print("R-squared (thực tiễn):", r2_test)

# Tạo DataFrame cho tháng 13
month_13 = pd.DataFrame({'Month': [13]})

# Chuẩn hóa dữ liệu cho tháng 13
month_13_scaled = scaler.transform(month_13)

# Dự đoán doanh thu tháng 13
predicted_sales_13 = model.predict(month_13_scaled)[0]

# In kết quả dự đoán
print("Dự đoán doanh thu tháng 13:", predicted_sales_13)