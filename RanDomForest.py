import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu từ file CSV
car_data = pd.read_csv('Car Price Prediction.csv')

# Làm sạch dữ liệu
# 1. Xử lý cột Levy
car_data['Levy'] = car_data['Levy'].replace('-', np.nan).astype(float)
car_data['Levy'] = car_data['Levy'].fillna(car_data['Levy'].median())  # Thay giá trị thiếu bằng trung vị

# 2. Chuyển đổi Mileage sang kiểu số
car_data['Mileage'] = car_data['Mileage'].str.replace(' km', '').str.replace(',', '').astype(float)

# 3. Tách thông tin Turbo từ Engine volume
car_data['Turbo'] = car_data['Engine volume'].str.contains('Turbo', na=False).astype(int)
car_data['Engine volume'] = car_data['Engine volume'].str.extract(r'([\d.]+)').astype(float)

# 4. Chuẩn hóa số cửa
car_data['Doors'] = car_data['Doors'].str.extract(r'(\d)').astype(float)

# 5. Xử lý target (Price)
car_data['Price'] = car_data['Price'].replace('-', np.nan).astype(float)
car_data.dropna(subset=['Price'], inplace=True)  # Xóa dòng thiếu giá trị Price

# 6. Log-transform giá trị Price
car_data['Price'] = np.log1p(car_data['Price'])

# Lựa chọn các cột quan trọng
selected_columns = [
    'Manufacturer', 'Levy', 'Prod. year', 'Mileage', 'Cylinders',
    'Engine volume', 'Turbo', 'Fuel type', 'Gear box type',
    'Drive wheels', 'Category', 'Leather interior', 'Airbags'
]
data = car_data[selected_columns]
target = car_data['Price']

# Mã hóa các cột phân loại
categorical_features = ['Manufacturer', 'Fuel type', 'Gear box type', 'Drive wheels', 'Category', 'Leather interior']
encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
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

# Xây dựng mô hình Random Forest
forest_model = RandomForestRegressor(random_state=42, n_estimators=100)
forest_model.fit(X_train, y_train)
y_pred_forest = forest_model.predict(X_test)

# Đánh giá mô hình
mse = mean_squared_error(y_test, y_pred_forest)
r2 = r2_score(y_test, y_pred_forest)

print("Random Forest MSE:", mse)
print("Random Forest R^2:", r2)

# Tầm quan trọng của đặc trưng
feature_names = numerical_features + list(encoder.get_feature_names_out(categorical_features))
feature_importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': forest_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Hiển thị tầm quan trọng
print(feature_importances)

# Trực quan hóa tầm quan trọng của đặc trưng
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances, color='skyblue')
plt.title("Feature Importance (Random Forest)")
plt.show()

# So sánh giá trị thực tế và dự đoán
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_forest, alpha=0.7, label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal Fit')
plt.xlabel('Actual Prices (log-transformed)')
plt.ylabel('Predicted Prices (log-transformed)')
plt.title('Actual vs Predicted Prices (Random Forest)')
plt.legend()
plt.show()

# Chuyển đổi lại giá trị dự đoán về giá trị thực
y_test_original = np.expm1(y_test)
y_pred_original = np.expm1(y_pred_forest)

# Trực quan hóa giá trị thực tế và dự đoán (thực)
plt.figure(figsize=(8, 6))
plt.scatter(y_test_original, y_pred_original, alpha=0.7, label='Predictions')
plt.plot([y_test_original.min(), y_test_original.max()], [y_test_original.min(), y_test_original.max()], 'r--', label='Ideal Fit')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices (Original Scale)')
plt.legend()
plt.show()
