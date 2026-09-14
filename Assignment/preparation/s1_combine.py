from pathlib import Path
import pandas as pd

base_dir = Path(__file__).resolve().parent.parent
data_dir = base_dir / "data"

df1 = pd.read_csv(data_dir / "customer_churn_dataset-testing-master.csv")
df2 = pd.read_csv(data_dir / "customer_churn_dataset-training-master.csv")

# Combine the two dataframes into a single dataframe
df = pd.concat([df1, df2], ignore_index=True)

print("Combined dataframe shape:", df.shape)

# Save the combined dataframe to the data folder
df.to_csv(data_dir / "full_churn_dataset.csv", index=False)