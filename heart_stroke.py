import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# Load the dataset
variable = pd.read_csv('/content/Stroke.csv')
print(variable)

# Extract the 'Age' column as features (X) and the last column as the target (y)
x = variable[['Age']]
y = variable.iloc[:, -1]
print(x)
print(y)
print(variable.columns)

# Handle missing values by replacing them with the mean value of the column
imputer = SimpleImputer(strategy='mean')
x_imputed = imputer.fit_transform(x)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_imputed, y, test_size=0.2, random_state=42)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# Initialize and train the Logistic Regression model
alg = LogisticRegression()
alg.fit(x_train, y_train)

# Retrieve the model's coefficients and intercept for the decision boundary
m = alg.coef_[0][0]
c = alg.intercept_[0]

# Generate a range of values for plotting the decision boundary
x_line = np.arange(x_train.min(), x_train.max(), 0.1)
# Apply the logistic sigmoid function to generate the corresponding y values
y_line = 1 / (1 + np.exp(-(m * x_line + c)))  # logistic sigmoid function

# Plot the decision boundary
plt.plot(x_line, y_line, label='Logistic Regression Boundary')
# Optionally, scatter plot the training data
# plt.scatter(x_train, y_train, label='Training Data', edgecolors='k', marker='o')
plt.xlabel('Age')
plt.ylabel('Probability')
plt.title('Logistic Regression Decision Boundary')
plt.legend()
plt.show()

# Print the training and testing scores
print('Training Score:', alg.score(x_train, y_train))
print('Testing Score:', alg.score(x_test, y_test))
