import matplotlib
import pandas as pd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

column_names = [0 for i in range(785)]
column_names[0] = 'class'
for i in range(1, 785):
    column_names[i] = "pixel" + str(i)
df_train = pd.read_csv('mnist_train.csv', header=None, names=column_names)
df_test = pd.read_csv('mnist_test.csv', header=None, names=column_names)


X_train = df_train.iloc[:, 1:785]
y_train = df_train.iloc[:, 0]
X_test = df_test.iloc[:, 1:785]
y_test = df_test.iloc[:, 0]

print(X_train.shape)
print(X_test.shape)

# wyznaczyć rozkład kategorii (w procentach)
train_counts = y_train.value_counts().sort_index()
category_dis_train = ((train_counts / train_counts.sum()) * 100)
print("Train category distribution percentage:\n", category_dis_train)
test_counts = y_test.value_counts().sort_index()
category_dis_test = ((test_counts / test_counts.sum()) * 100)
print("Test category distribution percentage:\n", category_dis_test)

# Narysowanie posortowanego wykresu słupkowego dla rozkładu kategorii zbioru treningowego
plt.figure(figsize=(10, 6))
train_counts.sort_index().plot(kind='bar')
plt.title('Posortowany rozkład kategorii dla zbioru treningowego MNIST')
plt.xlabel('Klasa cyfry')
plt.ylabel('Procent')
plt.xticks(rotation=0)
plt.show()
# Narysowanie posortowanego wykresu słupkowego dla rozkładu kategorii zbioru testowego
plt.figure(figsize=(10, 6))
test_counts.plot(kind='bar', color='orange')
plt.title('Posortowany rozkład kategorii dla zbioru testowego MNIST')
plt.xlabel('Klasa cyfry')
plt.ylabel('Procent')
plt.xticks(rotation=0)
plt.show()

for i in range(8):
     # define subplot
    plt.subplot(240+1+i)
    # plot raw pixel data
    ith_image = X_train.iloc[i, :]
    ith_image_arr = ith_image.to_numpy()
    ith_image= ith_image_arr.reshape(28, 28)
    plt.imshow(ith_image, cmap=plt.get_cmap('gray'))
plt.show()

for i in range(8):
     # define subplot
    plt.subplot(240+1+i)
    # plot raw pixel data
    ith_image = X_test.iloc[i, :]
    ith_image_arr = ith_image.to_numpy()
    ith_image= ith_image_arr.reshape(28, 28)
    plt.imshow(ith_image, cmap=plt.get_cmap('gray'))
plt.show()