import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 50486

ages = np.random.randint(18, 75, size=n_samples)
sexes = np.random.choice(["male", "female"], size=n_samples)
jobs = np.random.choice([0, 1, 2, 3], size=n_samples)
housing = np.random.choice(["own", "free", "rent"], size=n_samples)
saving_accounts = np.random.choice(["little", "moderate", "quite rich", "rich", np.nan], size=n_samples)
checking_accounts = np.random.choice(["little", "moderate", "rich", np.nan], size=n_samples)
credit_amounts = np.random.randint(200, 20000, size=n_samples)
durations = np.random.randint(6, 60, size=n_samples)
purposes = np.random.choice([
    "radio/TV", "education", "furniture/equipment", "car",
    "business", "domestic appliances", "repairs"
], size=n_samples)

data = pd.DataFrame({
    "Age": ages,
    "Sex": sexes,
    "Job": jobs,
    "Housing": housing,
    "Saving accounts": saving_accounts,
    "Checking account": checking_accounts,
    "Credit amount": credit_amounts,
    "Duration": durations,
    "Purpose": purposes
})

data.to_csv("synthetic_german_credit_data.csv", index_label="id")
print("✅ Synthetic data saved as 'synthetic_german_credit_data.csv'")
