import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

def load_dataset(filenames):
    for file in filenames:
        try:
            print(f"📂 Loading dataset: {file}")
            return pd.read_csv(file)
        except FileNotFoundError:
            print(f"❌ File not found: {file}")
    print("🚫 None of the datasets were found.")
    exit()

df = load_dataset(["synthetic_german_credit_data.csv", "german_credit_data.csv"])

df.drop(columns=[col for col in ["Unnamed: 0", "Job"] if col in df.columns], inplace=True)

cat_cols = df.select_dtypes(include='object').columns
num_cols = df.select_dtypes(include=[np.number]).columns

df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])
df[num_cols] = SimpleImputer(strategy="mean").fit_transform(df[num_cols])

np.random.seed(42)
saving_little = (df["Saving accounts"] == "little").astype(int)
checking_little = (df["Checking account"] == "little").astype(int)

prob_default = (
    0.2 * (df["Credit amount"] / df["Credit amount"].max()) +
    0.2 * (df["Duration"] / df["Duration"].max()) +
    0.3 * saving_little +
    0.3 * checking_little
)
df["Default"] = (np.random.rand(len(df)) < prob_default).astype(int)

X = df.drop(columns=["Default"])
y = df["Default"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.21, random_state=42)

categorical = X.select_dtypes(include='object').columns.tolist()
numeric = X.select_dtypes(include=np.number).columns.tolist()

preprocessor = ColumnTransformer(transformers=[ 
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
    ('num', 'passthrough', numeric)
])

pipeline = Pipeline(steps=[ 
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy on Test Set: {accuracy * 100:.2f}%")

def get_user_input():
    try:
        age = int(input("Enter Age: "))
        sex = input("Enter Sex (male/female): ").strip().lower()
        housing = input("Enter Housing (own/free/rent): ").strip().lower()
        saving_accounts = input("Enter Saving accounts (little/moderate/quite rich/rich): ").strip().lower()
        checking_account = input("Enter Checking account (little/moderate/rich): ").strip().lower()
        credit_amount = int(input("Enter Credit Amount: "))
        duration = int(input("Enter Duration (in months): "))
        purpose = input("Enter Purpose (radio/TV, education, furniture/equipment, car, business, domestic appliances, repairs): ").strip().lower()

        user_data = pd.DataFrame([{
            "Age": age,
            "Sex": sex,
            "Housing": housing,
            "Saving accounts": saving_accounts,
            "Checking account": checking_account,
            "Credit amount": credit_amount,
            "Duration": duration,
            "Purpose": purpose
        }])

        missing_columns = set(X.columns) - set(user_data.columns)
        if missing_columns:
            print(f"⚠️ Missing columns in user input: {missing_columns}")
            
            for col in missing_columns:
                if col in categorical:
                    user_data[col] = "None"  
                else:
                    user_data[col] = 0  

        return user_data
    except ValueError:
        print("⚠️ Invalid input. Please enter valid values.")
        exit()

user_df = get_user_input()

user_processed = pipeline.named_steps['preprocessor'].transform(user_df)
prediction = pipeline.named_steps['classifier'].predict(user_processed)[0]

user_df["Loan Eligibility"] = "Yes" if prediction == 0 else "No"

if prediction == 0:
    print("\n✅ You are eligible for the loan.")
else:
    print("\n❌ Sorry, you are not eligible for the loan.")

file_path = "user_input_result.csv"
if os.path.exists(file_path):
    user_df.to_csv(file_path, mode='a', header=False, index=False)
else:
    user_df.to_csv(file_path, mode='w', header=True, index=False)

print("\n✅ User input result has been saved to 'user_input_result.csv'.")

df['Loan Eligibility'] = df['Default'].apply(lambda x: 'Yes' if x == 0 else 'No')
df.to_csv("training_data_with_loan_eligibility.csv", index=False)
print("\n✅ Training data with loan eligibility has been saved to 'training_data_with_loan_eligibility.csv'.")
