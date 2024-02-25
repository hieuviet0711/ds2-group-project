import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = '/Users/dylan/Downloads/loan_data.csv'
loan_data = pd.read_csv(file_path)

# Substitute missing values using a summarization method: mode
for column in loan_data.columns:
    if loan_data[column].dtype == 'object':
        loan_data[column].fillna(loan_data[column].mode()[0], inplace=True)
    else:
        loan_data[column].fillna(loan_data[column].median(), inplace=True)

# Remove outliers using IQR
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

loan_data_cleaned = remove_outliers(loan_data, 'ApplicantIncome')
loan_data_cleaned = remove_outliers(loan_data_cleaned, 'CoapplicantIncome')
loan_data_cleaned = remove_outliers(loan_data_cleaned, 'LoanAmount')

# Set up the plot
plt.figure(figsize=(14, 8))

# Plot different relationships
sns.set(style="whitegrid")

# Plot distributions of numerical variables

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.histplot(loan_data_cleaned['ApplicantIncome'], kde=True, ax=axes[0, 0])

axes[0, 0].set_title('Applicant Income Distribution')

sns.histplot(loan_data_cleaned['CoapplicantIncome'], kde=True, ax=axes[0, 1])

axes[0, 1].set_title('Coapplicant Income Distribution')

sns.histplot(loan_data_cleaned['LoanAmount'], kde=True, ax=axes[1, 0])

axes[1, 0].set_title('Loan Amount Distribution')

sns.histplot(loan_data_cleaned['Loan_Amount_Term'], kde=True, ax=axes[1, 1])

axes[1, 1].set_title('Loan Amount Term Distribution')

plt.tight_layout()

plt.show()

# Education and ApplicantIncome
plt.subplot(2, 2, 1)
sns.boxplot(x='Education', y='ApplicantIncome', data=loan_data_cleaned)
plt.title('Education vs ApplicantIncome')

# Married and Dependents
plt.subplot(2, 2, 2)
sns.countplot(x='Married', hue='Dependents', data=loan_data_cleaned)
plt.title('Married vs Dependents')

# Property_Area and LoanAmount
plt.subplot(2, 2, 3)
sns.boxplot(x='Property_Area', y='LoanAmount', data=loan_data_cleaned)
plt.title('Property_Area vs LoanAmount')

# ApplicantIncome and Credit_History
plt.subplot(2, 2, 4)
sns.boxplot(x='Credit_History', y='ApplicantIncome', data=loan_data_cleaned)
plt.title('ApplicantIncome vs Credit_History')

plt.tight_layout()
plt.show()

# More relationship plots
plt.figure(figsize=(14, 8))

# Gender and Loan Status
plt.subplot(2, 2, 1)
sns.countplot(x='Gender', hue='Loan_Status', data=loan_data_cleaned)
plt.title('Gender vs Loan Status')

# Self_Employed and Loan Status
plt.subplot(2, 2, 2)
sns.countplot(x='Self_Employed', hue='Loan_Status', data=loan_data_cleaned)
plt.title('Self_Employed vs Loan Status')

# Credit_History and Loan Status
plt.subplot(2, 2, 3)
sns.countplot(x='Credit_History', hue='Loan_Status', data=loan_data_cleaned)
plt.title('Credit_History vs Loan Status')

# Property_Area and Loan Status
plt.subplot(2, 2, 4)
sns.countplot(x='Property_Area', hue='Loan_Status', data=loan_data_cleaned)
plt.title('Property_Area vs Loan Status')

plt.tight_layout()
plt.show()

# Set up the matplotlib figure

fig, axes = plt.subplots(4, 2, figsize=(14, 20))

# Education and ApplicantIncome

sns.boxplot(x='Education', y='ApplicantIncome', data=loan_data_cleaned, ax=axes[0, 0])

axes[0, 0].set_title('Education vs ApplicantIncome')

# Education and Credit_History

sns.countplot(x='Education', hue='Credit_History', data=loan_data_cleaned, ax=axes[0, 1])

axes[0, 1].set_title('Education vs Credit_History')

# Married and Dependents

sns.countplot(x='Married', hue='Dependents', data=loan_data_cleaned, ax=axes[1, 0])

axes[1, 0].set_title('Married vs Dependents')

# Married and Property_Area

sns.countplot(x='Married', hue='Property_Area', data=loan_data_cleaned, ax=axes[1, 1])

axes[1, 1].set_title('Married vs Property_Area')

# Married and CoapplicantIncome

sns.boxplot(x='Married', y='CoapplicantIncome', data=loan_data_cleaned, ax=axes[2, 0])

axes[2, 0].set_title('Married vs CoapplicantIncome')

# Property_Area and LoanAmount

sns.boxplot(x='Property_Area', y='LoanAmount', data=loan_data_cleaned, ax=axes[2, 1])

axes[2, 1].set_title('Property_Area vs LoanAmount')

# LoanAmount and Loan_Amount_Term

sns.scatterplot(x='LoanAmount', y='Loan_Amount_Term', data=loan_data_cleaned, ax=axes[3, 0])

axes[3, 0].set_title('LoanAmount vs Loan_Amount_Term')

# ApplicantIncome and Credit_History

sns.boxplot(x='Credit_History', y='ApplicantIncome', data=loan_data_cleaned, ax=axes[3, 1])

axes[3, 1].set_title('ApplicantIncome vs Credit_History')

plt.tight_layout()

plt.show()

