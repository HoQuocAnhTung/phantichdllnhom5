import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Đọc dữ liệu từ file "monthly_saless.csv"
df_sales = pd.read_csv("monthly_saless.csv")

# 2. Tạo một từ điển để lưu trữ mô hình hồi quy cho từng sản phẩm
models = {}

# 3. Lặp qua từng sản phẩm trong df_sales
for product in df_sales['Product'].unique():
    # Lấy dữ liệu của sản phẩm hiện tại
    df_product = df_sales[df_sales['Product'] == product]

    # 4. Chia dữ liệu thành tập train và test
    X_train, X_test, y_train, y_test = train_test_split(
        df_product[['Month']],
        df_product['Sales'],
        test_size=0.2,
        random_state=42
    )

    # 5. Chuẩn hóa dữ liệu
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 6. Huấn luyện mô hình hồi quy tuyến tính
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 7. Lưu mô hình vào từ điển
    models[product] = model

# 8. Tạo một danh sách để lưu trữ kết quả
all_results = []

# 9. Lặp qua từng sản phẩm trong models
for product, model in models.items():
    # 10. Dự đoán doanh thu tháng 13
    month_13 = pd.DataFrame({'Month': [13]})  # Tạo DataFrame cho month_13
    month_13_scaled = scaler.transform(month_13)  # Chuẩn hóa dữ liệu dự đoán
    predicted_sales_13 = model.predict(month_13_scaled)[0]

    # 11. Tính toán MSE và R-squared trên tập train
    y_train_pred = model.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)

    # 12. Tính toán MSE và R-squared trên tập test
    y_test_pred = model.predict(X_test)
    mse_test = mean_squared_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)

    # 13. Lưu kết quả vào danh sách
    all_results.append({
        'Product': product,
        'Predicted Sales (Month 13)': predicted_sales_13,
        'MSE (Train)': mse_train,
        'R2 (Train)': r2_train,
        'MSE (Test)': mse_test,
        'R2 (Test)': r2_test
    })

# 14. Tạo DataFrame từ danh sách kết quả
results = pd.DataFrame(all_results)

# 15. In DataFrame results
print(results.to_markdown(index=False, numalign="left", stralign="left"))

