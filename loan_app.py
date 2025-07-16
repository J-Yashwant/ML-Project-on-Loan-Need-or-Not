import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

@st.cache_data
def load_dataset():
    for file in ["synthetic_german_credit_data.csv", "german_credit_data.csv"]:
        try:
            return pd.read_csv(file)
        except FileNotFoundError:
            continue
    st.error("Dataset not found.")
    return None

df = load_dataset()
if df is None:
    st.stop()

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

st.title("🏦 Loan Eligibility Predictor")
st.write("Fill the form below to check your loan eligibility.")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
sex = st.selectbox("Sex", ["male", "female"])
housing = st.selectbox("Housing", ["own", "free", "rent"])
saving_accounts = st.selectbox("Saving Accounts", ["None","little", "moderate", "quite rich", "rich"])
checking_account = st.selectbox("Checking Account", ["None","little", "moderate", "rich"])
credit_amount = st.number_input("Credit Amount", min_value=100, max_value=100000, value=2000)
duration = st.number_input("Duration (months)", min_value=6, max_value=72, value=12)
purpose = st.selectbox("Purpose", ["radio/TV", "education", "furniture/equipment", "car", "business", "domestic appliances", "repairs"])

if st.button("Check Eligibility"):
    user_data = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "Housing": housing,
        "Saving accounts": saving_accounts if saving_accounts != "None" else saving_accounts,
        "Checking account": checking_account if checking_account != "None" else checking_account,
        "Credit amount": credit_amount,
        "Duration": duration,
        "Purpose": purpose
    }])

    for col in X.columns:
     if col not in user_data.columns:
        user_data[col] = np.nan if col in categorical else 0

    user_data = user_data[X.columns]

    user_processed = pipeline.named_steps["preprocessor"].transform(user_data)
    prediction = pipeline.named_steps["classifier"].predict(user_processed)[0]

    if prediction == 0:
        st.success("✅ You are eligible for the loan.")
    else:
        st.error("❌ Sorry, you are not eligible for the loan.")

#To Run in terminal:
#streamlit run "c:/Users/ya661/OneDrive/Desktop/4 TH SEM LAB/ML lab/ML Project/loan_app.py"