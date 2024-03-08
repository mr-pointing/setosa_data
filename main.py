import pandas as pd
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt

iris = load_iris()

# Displays the different parts of the data set
# print(dir(iris))

# Displays the different feature names we will need for our table
# print(iris.feature_names)

# Sets up our Dataframe with the data part and the feature names part
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.head())

# Appends target column into our Dataframe
df['target'] = iris.target
print(df.head())

# Prints the different target names
print(iris.target_names)

# Prints the header for target values equal to 1 and 2
print(df[df.target == 2].head())


# Add flower name to make table clearer using apply function
# lambda is a small function to apply a transformation
df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])
print(df.head())

# Seperate our different flowers into different dataframes
df0 = df[df.target == 0]
df1 = df[df.target == 1]
df2 = df[df.target == 2]

# Create a scatter plot to display sepal data
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'], color='green', marker="+")
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'], color='blue', marker=".")
plt.scatter(df2['sepal length (cm)'], df2['sepal width (cm)'], color='red', marker=".")
plt.show()
