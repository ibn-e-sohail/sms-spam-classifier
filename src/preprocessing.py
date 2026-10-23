import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure NLTK data is downloaded (good practice for production scripts)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(message: str) -> str:
    """Cleans a single SMS message string."""
    # 1. Remove special characters and numbers
    message = re.sub(r'[^a-zA-Z]', ' ', message)
    # 2. Convert to lowercase
    message = message.lower()
    # 3. Tokenize
    words = nltk.word_tokenize(message)
    # 4. Remove stop words and apply stemming
    cleaned_words = [ps.stem(word) for word in words if word not in stop_words]
    # 5. Join back to string
    return " ".join(cleaned_words)

def preprocess_dataframe(df):
    """Applies cleaning to the 'message' column of a pandas DataFrame."""
    df['cleaned_message'] = df['message'].apply(clean_text)
    return df