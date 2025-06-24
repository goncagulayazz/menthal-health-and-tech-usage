import pandas as pd
df = pd.read_csv("survey.csv")
print(df.head())

print(df.info())

# Mental health treatment received? (dependent variable)
df = df[df['treatment'].isin(['Yes', 'No'])]  # Only keep yes/no responses

# Some of the tech-related columns we want to observe:
selected_cols = ['treatment', 'tech_company', 'remote_work', 'benefits', 'family_history', 'work_interfere', 'care_options']

df = df[selected_cols].dropna()  # Remove missing values

# Convert categorical values to numerical
from sklearn.preprocessing import LabelEncoder

for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Separate dependent and independent variables
X = df.drop('treatment', axis=1)
y = df['treatment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction and evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Which variable had the most influence?
coeff_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

print(coeff_df)
