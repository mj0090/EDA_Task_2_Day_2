import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Creating the dataframe
df = pd.read_csv("cars_dataset.csv")

# Summary statistics
print(df.describe())
print(df.info())
print(df.isnull().sum())  # Check missing values

# Histograms for distribution
df.hist(bins=30, figsize=(12, 10))
plt.suptitle('Histograms of Numeric Features')
plt.show()

# Boxplots to detect outliers
for column in df.select_dtypes(include='number').columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x=column)
    plt.title(f'Boxplot of {column}')
    plt.show()

# Pairplot and Correlation Matrix

sns.pairplot(df.select_dtypes(include='number'))
plt.show()

# Correlation Heatmap

corr = df.corr(numeric_only=True)
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

print(df.skew(numeric_only=True))
# Apply log1p (log(1 + x)) to avoid issues with zero
df['price_log'] = np.log1p(df['price'])
df['mileage_log'] = np.log1p(df['mileage'])
df['tax_log'] = np.log1p(df['tax'])
df['mpg_log'] = np.log1p(df['mpg'])

# Visual check after transformation
sns.histplot(df['price_log'], kde=True)
plt.title("Log-Transformed Price")
plt.show()

# Check correlation
def drop_highly_correlated(df, threshold=0.8):
    corr_matrix = df.corr(numeric_only=True)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = [column for column in upper.columns if any(abs(upper[column]) > threshold)]
    return df.drop(columns=to_drop)

# Applying the function
df_cleaned = drop_highly_correlated(df)

# Removing outliers using IQR
Q1 = df['mileage'].quantile(0.25)
Q3 = df['mileage'].quantile(0.75)
IQR = Q3 - Q1

# Keep only data within IQR range (1.5x)
df = df[(df['mileage'] >= Q1 - 1.5 * IQR) & (df['mileage'] <= Q3 + 1.5 * IQR)]
