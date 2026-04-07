import sys
import pickle
import json
from cleaner import clean_text

def run_prediction():
    try:
        # 1. Load the pre-trained "brains"
        model = pickle.load(open('model.pkl', 'rb'))
        tfidf = pickle.load(open('tfidf.pkl', 'rb'))

        # 2. Get text from Node.js (passed as a command line argument)
        raw_text = sys.argv[1]
        
        # 3. Process and Predict
        cleaned = clean_text(raw_text)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)

        # 4. Return result as JSON string for Node to parse
        print(json.dumps({"result": str(prediction[0])}))
    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    run_prediction()