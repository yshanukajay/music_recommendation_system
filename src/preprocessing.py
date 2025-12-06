import pandas as pd
import re
import nltk
import joblib
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(
    level = logging.INFO,
    format = '[%(asctime)s] %(levelname)s - %(message)s',
    handlers = [
        logging.FileHandler('preprocess.log', encoding = 'utf-8'),
        logging.StreamHandler()
    ]
)

logging.info("🚀 Starting preprocessing...")

# Download ALL required NLTK data
try:
    nltk.download('punkt')
    nltk.download('punkt_tab')  # This is the missing resource
    nltk.download('stopwords')
    logging.info("✅ NLTK data downloaded successfully")
except Exception as e:
    logging.warning(f"NLTK download warning: {str(e)}")

try:
    df = pd.read_csv('../dataset/spotify_millsongdata.csv').sample(10000)
    logging.info('Dataset loaded & sampled : %d rows', len(df))
except Exception as e:
    logging.error('Failed to load the dataset: %s', str(e))
    raise e

df1 = df.drop(columns = ['link'], errors = 'ignore').reset_index(drop = True)

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Handle NaN values
    if pd.isna(text):
        return ''
    
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))
    text = text.lower()
    
    # Simple tokenization without NLTK if needed
    try:
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words]
    except LookupError:
        # Fallback: simple split if NLTK fails
        tokens = text.split()
        tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

logging.info("🧹 Cleaning text...")
df1['cleaned_text'] = df1['text'].apply(preprocess_text)
logging.info("✅ Text cleaned.")

logging.info("🔠 Vectorizing using TF-IDF...")
tfidf = TfidfVectorizer(max_features = 5000)
tfidf_matrix = tfidf.fit_transform(df1['cleaned_text'])
logging.info("✅ TF-IDF matrix shape: %s", tfidf_matrix.shape)

logging.info("📐 Calculating cosine similarity...")
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
logging.info("✅ Cosine similarity matrix shape: %s", cosine_sim.shape)

logging.info("💾 Saving models...")
joblib.dump(df1, 'data_cleaned.pkl')
joblib.dump(tfidf_matrix, 'tfidf_matrix.pkl')
joblib.dump(cosine_sim, 'cosine_sim.pkl')
logging.info("✅ Models saved.")

logging.info('🎉 Preprocessing completed successfully!')