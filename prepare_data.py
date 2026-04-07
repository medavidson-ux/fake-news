import pandas as pd

# Load both datasets
# Note: Ensure you have 'True.csv' in your folder!
fake_df = pd.read_csv('Fake.csv', low_memory=False)
true_df = pd.read_csv('True.csv', low_memory=False)

# Assign clear labels
fake_df['label'] = 'FAKE'
true_df['label'] = 'REAL'

# Combine them into one big table
combined_df = pd.concat([fake_df, true_df]).reset_index(drop=True)

# Save the combined data for training
combined_df.to_csv('combined_news.csv', index=False)
print("Success! Created combined_news.csv with both Fake and Real examples.")