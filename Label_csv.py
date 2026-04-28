import pandas as pd

df = pd.read_csv(r"Panchayat Season 2 CSV\Episode_8.csv")  #change per file

# Map labels to 0/1
Labels = {"Humour": 1, "Not Humour": 0}
df['Labels'] = df['Humour?'].map(Labels).fillna(0).astype(int)


# Save the new CSV with labels
df.to_csv("Panchayat Season 2 CSV\Episode_8.csv", index=False)
