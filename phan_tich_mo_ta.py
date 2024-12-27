import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("all2.csv")

df1 = pd.DataFrame(df.dtypes, columns=['Data Type'])  # Chuyển đổi Series thành DataFrame
print(df1.describe(include="all"))
########1 ve bieu do cot the hien doanh thu theo thang
df['Quantity Ordered'] = pd.to_numeric(df['Quantity Ordered'], downcast='integer')
df['Price Each'] = pd.to_numeric(df['Price Each'], downcast='float')
df['Sales'] = df['Quantity Ordered'] * df['Price Each']
moving_column = df.pop('Sales')
df.insert(4, 'Sales', moving_column)
df.groupby('Month').sum()['Sales']
sales_value = df.groupby('Month').sum()['Sales']
print(sales_value)
months = range(1,13)
plt.bar(x=months, height=sales_value)
plt.xticks(months)
plt.xlabel('Months')
plt.ylabel('Sales in USD')
plt.show()

#####2 doanh thu theo tưng khu vuc

address_to_city = lambda address:address.split(',')[1]
df['City'] = df['Purchase Address'].apply(address_to_city)
print(df.head())
df.groupby('City').sum()['Sales']
sales_value_city = df.groupby('City').sum()['Sales']
cities = [city for city, sales in sales_value_city.items()]
plt.bar(x=cities, height=sales_value_city)
plt.xticks(cities, rotation=90, size=8)
plt.xlabel('Cities')
plt.ylabel('Sales in USD')
plt.show()


# Vẽ scatter plot1
plt.scatter(df['Order ID'], df['Price Each'], alpha=0.5)

# Tính toán đường hồi quy
m, b = np.polyfit(df['Order ID'], df['Price Each'], 1)
plt.plot(df['Order ID'], m*df['Order ID'] + b, color='red')

# Thêm các thông tin cho biểu đồ
plt.xlabel('Order ID')
plt.ylabel('Price Each')
plt.title('Scatter Plot with Regression Line')
plt.grid(True)
plt.show()

# Vẽ scatter plot2
plt.scatter(df['Quantity Ordered'], df['Price Each'], alpha=0.5)

# Tính toán đường hồi quy
m, b = np.polyfit(df['Quantity Ordered'], df['Price Each'], 1)
plt.plot(df['Quantity Ordered'], m*df['Order ID'] + b, color='red')

# Thêm các thông tin cho biểu đồ
plt.xlabel('Quantity Ordered')
plt.ylabel('Price Each')
plt.title('Scatter Plot with Regression Line')
plt.grid(True)
plt.show()

# Vẽ scatter plot3
plt.scatter(df['Month'], df['Price Each'], alpha=0.5)

# Tính toán đường hồi quy
m, b = np.polyfit(df['Month'], df['Price Each'], 1)
plt.plot(df['Month'], m*df['Month'] + b, color='red')

# Thêm các thông tin cho biểu đồ
plt.xlabel('Month')
plt.ylabel('Price Each')
plt.title('Scatter Plot with Regression Line')
plt.grid(True)
plt.show()

###### Ve bieu do boxplot
df_numeric = df.select_dtypes(include=['number'])

for column in df_numeric.columns:
    plt.figure(figsize=(6, 4))  # Kích thước của biểu đồ
    plt.boxplot(df[column])
    plt.title(f'Box Plot of {column}')
    plt.ylabel(column)
    plt.grid(True)
    plt.show()

######3 bieu do duong the hien thoi gian mau so luong don hang
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Hours'] = df['Order Date'].dt.hour
sales_value_hours = df.groupby('Hours').count()['Sales']
hours = [hour for hour, sales in sales_value_hours.items()]
plt.plot(hours, sales_value_hours)
plt.grid()
plt.xticks(hours, rotation=90, size=8)
plt.xlabel('Hours')
plt.ylabel('Sales in USD')
plt.show()

#####4 ve moi quan he giua doanh thu va

all_products = df.groupby('Product').sum()['Quantity Ordered']
prices = df.groupby('Product').mean()['Price Each']
products_ls = [product for product, quant in all_products.items()]

x = products_ls
y1 = all_products
y2 = prices

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.bar(x, y1, color='g')
ax2.plot(x, y2, 'b-')

ax1.set_xticklabels(products_ls, rotation=90, size=8)
ax1.set_xlabel('Products')
ax1.set_ylabel('Quantity Ordered', color='g')
ax2.set_ylabel('Price Each', color='b')