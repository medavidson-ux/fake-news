import sys
import pickle
import numpy as np

# 1. Load the existing brains
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# 2. Get the correction data
new_text = sys.argv[1]
true_label = sys.argv[2]

# 3. Transform and Update
vectorized_text = tfidf.transform([new_text])

# partial_fit requires the list of all possible classes
model.partial_fit(vectorized_text, [true_label], classes=['FAKE', 'REAL'])

# 4. Save the "smarter" brains back to disk
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf.pkl', 'wb'))