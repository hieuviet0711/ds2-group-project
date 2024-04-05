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
