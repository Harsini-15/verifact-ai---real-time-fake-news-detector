import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
def download_nltk_resources():
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('tokenizers/punkt_tab')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('wordnet')
        nltk.download('omw-1.4')

download_nltk_resources()

class NLPPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Convert to lowercase
        text = text.lower()
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords and lemmatize
        cleaned_tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words
        ]
        return " ".join(cleaned_tokens)

    def extract_keywords(self, text, top_n=5):
        # Simple frequency-based keyword extraction after cleaning
        cleaned = self.clean_text(text)
        tokens = cleaned.split()
        if not tokens:
            return []
        
        freq_dist = nltk.FreqDist(tokens)
        return [word for word, count in freq_dist.most_common(top_n)]
