# Necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
df = pd.read_csv('loan_data.csv')

# Remove rows with missing data
df.dropna(inplace=True)

# Drop unnecessary columns
df = df.drop(['Loan_ID', 'Gender'], axis=1)

# Change varibles with strings to int
df['Dependents'] = df['Dependents'].replace('3+', 3)
label_encoders = preprocessing.LabelEncoder()
for column in ['Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']:
    df[column] = label_encoders.fit_transform(df[column])
    df[column].unique()


# Remove outliers
def remove_outliers(dff, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

for column in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']:
    df = remove_outliers(df, column)

# Train Test Spilt
y = df['Loan_Status']
X = df.drop(['Loan_Status'], axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=5)

# Train SVM model on train set
model_svm = SVC(random_state=5)
model_svm.fit(X_train, y_train)
pred = model_svm.predict(X_test)
print(classification_report(y_test, pred))

# Hyperparameter Tuning for SVM
param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf', 'sigmoid', 'poly']}
grid = GridSearchCV(SVC(random_state=5), param_grid, refit=True, verbose=0)
grid = grid.fit(X_train, y_train)

print(grid.best_params_)
print("Best cross-validation score:", grid.best_score_)
cv_scores = cross_val_score(grid, X_train, y_train, cv=5, scoring='accuracy')
bias = 1 - cv_scores.mean()
variance = cv_scores.std()

print("Bias:", bias)
print("Variance:", variance)

grid_predictions = grid.predict(X_test)
print(classification_report(y_test, grid_predictions))


# blue is no, red is yes
# Perform PCA for dimensionality reduction
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)


model_svm_pca = SVC(random_state=5)
model_svm_pca.fit(X_train_pca, y_train)

def plot_decision_boundary(model, X, y):
    h = 0.02 
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    plt.xlabel('')
    plt.ylabel('')
    plt.title('Decision Boundary')

# Plot decision boundary for SVM with PCA
plt.figure(figsize=(8, 6))
plot_decision_boundary(model_svm_pca, X_train_pca, y_train)
plt.show()


print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


# Train Logistic Regression model on train set
model_logreg = LogisticRegression(random_state=5, max_iter=1000)
model_logreg.fit(X_train, y_train)
pred2 = model_logreg.predict(X_test)
print(classification_report(y_test, pred2))

# Hyperparameter Tuning for Logistic Regression
param_grid2 = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
grid2 = GridSearchCV(LogisticRegression(random_state=5, max_iter=1000), param_grid2, refit=True, verbose=0,)
grid2 = grid2.fit(X_train, y_train)

print(grid2.best_params_)
print("Best cross-validation score:", grid2.best_score_)
cv_scores = cross_val_score(grid2, X_train, y_train, cv=5, scoring='accuracy')
bias = 1 - cv_scores.mean()
variance = cv_scores.std()

print("Bias:", bias)
print("Variance:", variance)

grid_predictions2 = grid2.predict(X_test)
print(classification_report(y_test, grid_predictions2))

coefficients = model_logreg.coef_[0]
feature_names = X.columns
#coefficient data frame
coefficients_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

# Sort the coefficients by absolute value
coefficients_df['Abs_Coefficient'] = np.abs(coefficients_df['Coefficient'])
coefficients_df = coefficients_df.sort_values(by='Abs_Coefficient', ascending=False)

# Plot the coefficients
plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=coefficients_df)
plt.xlabel('Coefficient')
plt.ylabel('Feature')
plt.title('Logistic Regression Coefficients')
plt.show()

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


# Tain Decision Tree model on train set
model_dectree = DecisionTreeClassifier(random_state=5)
model_dectree.fit(X_train, y_train)
pred3 = model_dectree.predict(X_test)
print(classification_report(y_test, pred3))

# Hyperparameter Tuning for Decision Tree
param_grid3 = {'criterion': ['gini', 'entropy'], 'max_depth': [None, 5, 10, 15, 20], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
grid3 = GridSearchCV(DecisionTreeClassifier(random_state=5), param_grid3, refit=True, verbose=0)
grid3 = grid3.fit(X_train, y_train)

print(grid3.best_params_)
print("Best cross-validation score:", grid3.best_score_)
cv_scores = cross_val_score(grid3, X_train, y_train, cv=5, scoring='accuracy')
bias = 1 - cv_scores.mean()
variance = cv_scores.std()

print("Bias:", bias)
print("Variance:", variance)

grid_predictions3 = grid3.predict(X_test)
print(classification_report(y_test, grid_predictions3))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

#display decision tree
from sklearn.tree import plot_tree

model_dectree = DecisionTreeClassifier(random_state=5)
model_dectree.fit(X_train, y_train)

# Visualize the decision tree
plt.figure(figsize=(20,10))
plot_tree(model_dectree, filled=True, feature_names=X.columns, class_names=['No', 'Yes'])
plt.show()