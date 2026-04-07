import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize tools
nltk.download('stopwords', quiet=True)
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Remove non-letters, lowercase, and split
    review = re.sub('[^a-zA-Z]', ' ', str(text)).lower().split()
    # Stem words and remove stop words
    review = [ps.stem(word) for word in review if not word in stop_words]
    return ' '.join(review)