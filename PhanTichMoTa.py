import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
import pycountry as pct

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('Car Price Prediction.csv') #Để file dataset cùng 1 folder với file code
description = data.describe()
mode = data.select_dtypes(include=['float64','int64']).mode().iloc[0]
mode.name = 'mode'
median = data.select_dtypes(include=['float64','int64']).median()
median.name = 'median'
var = data.select_dtypes(include=['float64', 'int64']).var()
var.name = 'var'
iqr = data.select_dtypes(include=['float64', 'int64']).quantile(0.75) - data.select_dtypes(include=['float64', 'int64']).quantile(0.25)
iqr.name = 'iqr'
print(description)
# Kiểm tra tỷ lệ lỗi thiếu data
data_na = (data.isnull().sum() / len(data)) * 100
missing_data = pd.DataFrame({'Ty le thieu data': data_na})
print(missing_data)

# Kiểm tra data bị trùng
duplicated_rows_data = data.duplicated().sum()
print(f"\nSO LUONG DATA BI TRUNG LAP: {duplicated_rows_data}")
data = data.drop_duplicates()

# Quét qua các cột và đếm số lượng data riêng biệt
print("\nSO LUONG CAC DATA RIENG BIET:")
for column in data.columns:
    num_distinct_values = len(data[column].unique())
    print(f"{column}:{num_distinct_values} distinct values")

# Xem qua dataset
print(f"\n5 DONG DAU DATA SET:\n {data.head(5)}")


# Chuẩn hóa dữ liệu để phân khúc loại xe
def vehicle_segment(Price):
    if Price < 1000:
        return "Xe hạng thấp"
    elif Price >= 1000 and Price <= 800000:
        return "Xe bình dân"
    elif Price > 800000:
        return "Xe hạng sang"
    else:
        return "Other"

# Áp dụng hàm và tạo cột 'vehicle_segment'
data['Vehicle segment'] = data['Price'].apply(vehicle_segment)
print(data)
df =data
df.to_csv('after.csv')
column_data = df['Manufacturer']
print(column_data)

# dem so lan xuat hien cua cac phan tu
duplicate_counts = column_data.value_counts()
pd.set_option('display.max_rows', None)
print(duplicate_counts)

#tong hop cac brand có manufacture <250 và sum chung = other
small_brand = duplicate_counts[duplicate_counts<250]
other_count= small_brand.sum()
duplicate_counts = duplicate_counts[duplicate_counts >= 250]
duplicate_counts['Other_brand'] = other_count
# Vẽ đồ thị hình tròn
fig = px.pie(values=duplicate_counts.values, title="Car Manufacturer")
fig.update_layout(title_x=0.45)
fig.show()

column_price = df['Price']
print(df['Price'].describe())
# Áp dụng log-transform (thêm 1 để tránh log(0) nếu có giá trị bằng 0)
df['Log_Price'] = np.log(df['Price'] + 1)


print(column_price)

df_filtered = df[df['Price'] > 5000]
df_filtered = df_filtered.dropna(subset=['Price'])  # Loại bỏ các giá trị NaN trong cột Price
print(df_filtered['Price'].describe())  # Kiểm tra các thông số thống kê của Price sau khi lọc
df_filtered = df[df['Price'] > 5000]
df_filtered = df_filtered[df_filtered['Price'] < 100000]  # Lọc các giá trị quá cao và các giá trị quá thấp

df_filtered['Price'] = np.log(df_filtered['Price']) #tao 1 mang các giá trị đã chuyển đổi

plt.figure(figsize=(12, 12))  #tỉ lên màn khi chạy
sns.displot(df_filtered['Price'], kde=True, bins=30)  #chia dl thành 30 vùng
plt.title('Car_Price',fontsize=16, fontweight='bold')
plt.subplots_adjust(top=0.85)
plt.show()
#
# #3.Biểu đồ giá xe theo loại xe
df_filtered = df[(df['Price'] > 5000) & (df['Price'] < 100000)]
df_filtered = df_filtered.dropna(subset=['Price'])  # Loại bỏ các giá trị NaN

# Tăng kích thước biểu đồ
plt.figure(figsize=(16, 8))

# Tạo biểu đồ hộp với trục y ở thang logarit
sns.boxplot(data=df_filtered, x='Category', y='Price',color='green')
plt.yscale('log')  # Chuyển trục y sang thang logarit để mở rộng giá trị thấp

# Tùy chỉnh biểu đồ
plt.title('Biểu đồ giá xe theo loại xe (Vehicle Type)',fontsize=16,color='red')
plt.xlabel('Loại xe (Vehicle Type)')
plt.ylabel('Giá xe (Price) - Thang logarit')
plt.xticks(rotation=45)  # Xoay nhãn trên trục x để dễ đọc hơn

plt.show()
#
# #4.Biểu đồ giá xe trung bình theo năm sản xuất
df_filtered = df[(df['Price'] > 5000) & (df['Price'] < 100000)]
df_filtered = df_filtered.dropna(subset=['Price'])  # Loại bỏ các giá trị NaN
df_avg_price = df_filtered.groupby('Prod. year')['Price'].mean().reset_index()

# Tạo biểu đồ đường (Line Chart)
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_avg_price, x='Prod. year', y='Price', marker='o')

# Tùy chỉnh biểu đồ
plt.title('Biểu đồ giá xe trung bình theo năm sản xuất (Average Price by Production Year)',fontsize=16,color='red')
plt.xlabel('Năm sản xuất (Production Year)')
plt.ylabel('Giá xe trung bình (Average Price)')
plt.xticks(rotation=45)  # Xoay nhãn trục x nếu cần

plt.show()


#5


manufacturer_counts = df['Manufacturer'].value_counts()
valid_manufacturers = manufacturer_counts[manufacturer_counts >= 250].index

# Giữ lại chỉ những hãng xe hợp lệ trong dữ liệu
df_filtered = df[df['Manufacturer'].isin(valid_manufacturers)]


df_filled = df_filtered[(df_filtered['Price'] > 5000) & (df_filtered['Price'] < 100000)]
df_filled = df_filled.dropna(subset=['Price', 'Mileage', 'Manufacturer'])
df_filled_km = df_filled['Mileage']
print(f'in bảng {df_filled_km}')


# Tạo biểu đồ phân tán với Manufacturer làm nhóm phân biệt
plt.figure(figsize=(12, 6))  # Kích thước biểu đồ

df_sampled = df_filled.sample(frac=0.7, random_state=42)


# Vẽ biểu đồ phân tán với 'Mileage' trên trục x, 'Price' trên trục y, và phân nhóm theo 'Manufacturer'
sns.scatterplot(data=df_sampled, x='Mileage', y='Price', hue='Manufacturer', alpha=0.6)

# Tùy chỉnh biểu đồ
plt.title('Biểu đồ giá xe dựa trên số km đã đi và nhà sản xuất (Price vs Mileage vs Manufacturer)',fontsize=16,color='red')
plt.xlabel('Số km đã đi (Mileage)')  # Nhãn trục x
plt.ylabel('Giá xe (Price)')  # Nhãn trục y0

# Đặt mốc cho trục x (Số km đã đi) theo yêu cầu
plt.xticks(range(0,5001,500 ), ['0','20000','40000','60000','80000','100000','120000','140000','160000','180000','200000'])


plt.legend(title='Nhà sản xuất', bbox_to_anchor=(1.05, 1), loc='upper left')  # Đưa legend ra ngoài biểu đồ để không che dữ liệu

# Hiển thị biểu đồ
plt.show()
#6.
# Kiểm tra dữ liệu

# Loại bỏ "Turbo" và chỉ giữ lại các giá trị số
# Giả sử df là DataFrame của bạn
df['Engine volume'] = df['Engine volume'].str.replace('Turbo', '')  # Loại bỏ "Turbo"
df['Engine volume'] = pd.to_numeric(df['Engine volume'], errors='coerce')  # Chuyển đổi thành số

# Lọc dữ liệu trong khoảng giá từ 5000 đến 20000
df_filtered = df[(df['Price'] >= 5000) & (df['Price'] <= 20000)]

# Kiểm tra dữ liệu sau khi lọc
print("\nDữ liệu sau khi lọc:")
print(df_filtered[['Price', 'Engine volume']])

# Lấy mẫu ngẫu nhiên 10% dữ liệu
df_sample = df_filtered.sample(frac=0.1, random_state=42)

# Vẽ Boxplot với các nhóm dung tích động cơ
plt.figure(figsize=(10, 6))
sns.boxplot(x='Engine volume', y='Price', data=df_sample,color='lightgreen')  # Dùng df_sample đã lọc và giảm thiểu dữ liệu
plt.title('Biểu đồ giá xe theo dung tích động cơ (Price by Engine volume)', fontsize=16, color='red')
plt.xlabel('Dung tích động cơ (Engine volume)')
plt.ylabel('Giá xe (Price)')
plt.xticks(rotation=45)  # Xoay nhãn trục X cho dễ đọc
plt.show()