import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
import numpy as np


# 1. Đọc dữ liệu từ file "son123.csv"
df_sales = pd.read_csv("son123.csv")

# Hiển thị 5 hàng đầu tiên của df_sales
print(df_sales.head().to_markdown(index=False, numalign="left", stralign="left"))

# Hiển thị thông tin về các cột và kiểu dữ liệu của df_sales
print(df_sales.info())

# Lấy tất cả các giá trị duy nhất từ `Product`
unique_values = df_sales['Product'].unique()

# Kiểm tra số lượng các giá trị duy nhất trong `Product`
if len(unique_values) > 50:
    # Nếu có quá nhiều giá trị duy nhất, lấy mẫu 50 giá trị xuất hiện nhiều nhất
    top_occurring_values = df_sales['Product'].value_counts().head(50).index.tolist()
    print(top_occurring_values)
else:
    # Ngược lại, in tất cả các giá trị duy nhất trong `Product`
    print(unique_values)

# Tạo một danh sách để lưu trữ kết quả của mỗi mô hình
all_results = []

# Lặp qua từng sản phẩm
for product in df_sales['Product'].unique():
    # Tạo DataFrame mới chỉ chứa dữ liệu của sản phẩm hiện tại
    df_product = df_sales[df_sales['Product'] == product].copy()

    # Chia dữ liệu thành tập train và test (80/20)
    X = df_product[['Month']]
    y = df_product['Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Tạo đặc trưng theo chu kỳ cho tập train
    X_train['Month_sin'] = np.sin(2 * np.pi * X_train['Month'] / 12)
    X_train['Month_cos'] = np.cos(2 * np.pi * X_train['Month'] / 12)

    # Chuẩn hóa tập train và test
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Tạo đặc trưng theo chu kỳ cho tập test
    X_test['Month_sin'] = np.sin(2 * np.pi * X_test['Month'] / 12)
    X_test['Month_cos'] = np.cos(2 * np.pi * X_test['Month'] / 12)
    X_test_scaled = scaler.transform(X_test)

    # Tạo pipeline
    pipeline = Pipeline([
        ('selector', SelectKBest(score_func=f_regression, k=2)),
        ('model', LinearRegression())
    ])

    # Huấn luyện mô hình trên tập train
    pipeline.fit(X_train_scaled, y_train)

    # Dự đoán trên tập train và tính MSE, R2 thẩm định
    y_train_pred = pipeline.predict(X_train_scaled)
    mse_train = mean_squared_error(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)

    # Dự đoán trên tập test và tính MSE, R2 thực tiễn
    y_test_pred = pipeline.predict(X_test_scaled)
    mse_test = mean_squared_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)

    # Dự đoán cho tháng 13
    month_13 = pd.DataFrame({'Month': [13]})
    month_13['Month_sin'] = np.sin(2 * np.pi * month_13['Month'] / 12)
    month_13['Month_cos'] = np.cos(2 * np.pi * month_13['Month'] / 12)
    month_13_scaled = scaler.transform(month_13)
    predicted_sales_13 = pipeline.predict(month_13_scaled)[0]

    # In kết quả
    print(f"------Kết quả cho sản phẩm {product}-------")
    print("MSE (thẩm định):", mse_train)
    print("R-squared (thẩm định):", r2_train)
    print("MSE (thực tiễn):", mse_test)
    print("R-squared (thực tiễn):", r2_test)
    print("Dự đoán doanh thu tháng 13:", predicted_sales_13)

    # Lưu kết quả
    all_results.append({
        'Product': product,
        'Predicted Sales (Month 13)': predicted_sales_13,
        'MSE (Train)': mse_train,
        'R2 (Train)': r2_train,
        'MSE (Test)': mse_test,
        'R2 (Test)': r2_test
    })

# Chuyển đổi danh sách kết quả thành DataFrame
results_df = pd.DataFrame(all_results)

# Hiển thị DataFrame
print("\n------ Bảng tổng hợp kết quả ------")
print(results_df.to_markdown(index=False, numalign="left", stralign="left"))