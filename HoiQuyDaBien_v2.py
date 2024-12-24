import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV
car_data = pd.read_csv('car_price_prediction.csv')
# Tóm lược dữ liệu(Đo mức độ tập trung & mức độ phân tán)
description = car_data.describe()
mode = car_data.select_dtypes(include=['float64','int64']).mode().iloc[0]
mode.name = 'mode'
median = car_data.select_dtypes(include=['float64','int64']).median()
median.name = 'median'
var = car_data.select_dtypes(include=['float64', 'int64']).var()
var.name = 'var'
iqr = car_data.select_dtypes(include=['float64', 'int64']).quantile(0.75) - car_data.select_dtypes(include=['float64', 'int64']).quantile(0.25)
iqr.name = 'iqr'
description = description._append(mode)
description = description._append(median)
description = description._append(var)
description = description._append(iqr)
print(description)
print('-------------------------------------------------------------------------------------------------------------------')
# Làm sạch dữ liệu
# 1.1 Kiểm tra tỷ lệ lỗi thiếu data
data_na = (car_data.isnull().sum() / len(car_data)) * 100
missing_data = pd.DataFrame({'Ty le thieu data': data_na})
print(missing_data)

# 1.2 Kiểm tra data bị trùng
duplicated_rows_data = car_data.duplicated().sum()
print(f"\nSO LUONG DATA BI TRUNG LAP: {duplicated_rows_data}")
data = car_data.drop_duplicates()
print('-------------------------------------------------------------------------------------------------------------------')
# 1. Xử lý cột Levy
car_data['Levy'] = car_data['Levy'].replace('-', np.nan).astype(float)
car_data['Levy'] = car_data['Levy'].fillna(car_data['Levy'].median())  # Dùng giá trị trung vị

# 2. Chuyển đổi Mileage sang kiểu số
car_data['Mileage'] = car_data['Mileage'].str.replace(' km', '').str.replace(',', '').astype(float)

# 3. Tách thông tin Turbo từ Engine volume
car_data['Turbo'] = car_data['Engine volume'].str.contains('Turbo', na=False).astype(int)
car_data['Engine volume'] = car_data['Engine volume'].str.extract(r'([\d.]+)').astype(float)

# 4. Chuẩn hóa số cửa
car_data['Doors'] = car_data['Doors'].str.extract(r'(\d)').astype(float)
print(car_data.head(10))
print('-------------------------------------------------------------------------------------------------------------------')
# Lựa chọn các cột quan trọng
selected_columns = [
    'Manufacturer', 'Levy', 'Prod. year', 'Mileage', 'Cylinders',
    'Engine volume', 'Turbo', 'Fuel type', 'Gear box type',
    'Drive wheels', 'Category', 'Leather interior', 'Airbags'
]
data = car_data[selected_columns]

# Xử lý target (Price)
car_data['Price'] = car_data['Price'].replace('-', np.nan).astype(float)
car_data.dropna(subset=['Price'], inplace=True)  # Xóa dòng thiếu giá trị Price
target = car_data['Price']

# Mã hóa các cột phân loại
categorical_features = ['Manufacturer', 'Fuel type', 'Gear box type', 'Drive wheels', 'Category', 'Leather interior']
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_data = encoder.fit_transform(data[categorical_features])

# Gộp dữ liệu sau mã hóa với dữ liệu số
numerical_features = ['Levy', 'Prod. year', 'Mileage', 'Cylinders', 'Engine volume', 'Turbo', 'Airbags']
numerical_data = data[numerical_features].values

final_data = np.hstack([numerical_data, encoded_data])

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
final_data = scaler.fit_transform(final_data)

# Tách dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(final_data, target, test_size=0.2, random_state=42)

# Xây dựng mô hình hồi quy tuyến tính đa biến
multi_model = LinearRegression()
multi_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = multi_model.predict(X_test)

# Đánh giá mô hình
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R^2 Score:", r2)

# Lấy tên các đặc trưng
feature_names = numerical_features + list(encoder.get_feature_names_out(categorical_features))

# Hệ số hồi quy
coefficients = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': multi_model.coef_
}).sort_values(by='Coefficient', ascending=False)

# Hiển thị kết quả
print(coefficients)

# Trực quan hóa hệ số hồi quy
plt.figure(figsize=(10, 6))
plt.barh(coefficients['Feature'], coefficients['Coefficient'], color='skyblue')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.title('Feature Importance in Multivariable Regression')
plt.gca().invert_yaxis()
plt.show()
