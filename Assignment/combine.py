import pandas as pd

df1 = pd.read_csv("C:\\Users\\User\\ODL_Labs\\Assignment\\customer_churn_dataset-testing-master.csv")
df2 = pd.read_csv("C:\\Users\\User\\ODL_Labs\\Assignment\\customer_churn_dataset-training-master.csv")

# Combine the two dataframes into a single dataframe
df = pd.concat([df1, df2], ignore_index=True)

print("Combined dataframe shape:", df.shape)

# Save the combined dataframe to a new CSV file
df.to_csv("C:\\Users\\User\\ODL_Labs\\Assignment\\full_churn_dataset.csv", index=False)