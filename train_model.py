import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle

# 1. Load Dataset with low_memory=False to handle the column warning
# We only load the columns we actually need to save memory and avoid errors
try:
    # Adjust 'text' and 'label' if your CSV column names are different
    df = pd.read_csv('combined_news.csv', low_memory=False)
    
    # 2. Data Cleaning: Crucial Fixes
    # Remove any rows where 'text' or 'label' is missing
    df.dropna(subset=['text', 'label'], inplace=True)
    
    # Ensure everything in the text column is a string (prevents the 'np.nan' error)
    df['text'] = df['text'].astype(str)
    
    labels = df.label
except KeyError:
    print("Error: Could not find columns named 'text' or 'label'. Check your CSV header.")
    exit()

# 3. Split data
x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)

# 4. Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(x_train)

# 5. Train Model
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# 6. Save the brains
pickle.dump(pac, open('model.pkl', 'wb'))
pickle.dump(tfidf_vectorizer, open('tfidf.pkl', 'wb'))

print("Success! model.pkl and tfidf.pkl have been created.")