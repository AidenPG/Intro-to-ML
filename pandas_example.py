import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length', 'petal width', 'class']
df = pd.read_csv(url, header = None, names=column_names)

#Save the dataframe to local csv file

df.to_csv('iris_saved.csv', index=False)
df = pd.read_csv('iris_saved.csv')

print("Shape of the Dataframe: ", df.shape)
print("\nFirst 5 rows of the Dataframe: ")
print(df.head())

print("\nData types of each column: ")
print(df.dtypes)

print("\nSummary statistics of the dataframe: ")
print(df.describe())

mean_sepal_length = df['sepal length (cm)'].mean()
median_sepal_length = df['sepal length (cm)'].median()
std_petal_length = df['sepal length (cm)'].std()

grouped_mean = df.groupby('class').mean()
grouped_stats = df.groupby('class').agg(['mean','std', 'min','max'])

class_count = df['class'].value_counts()
missing_values = df.isnull().sum()
df.drop_duplicates(inplace=True)


