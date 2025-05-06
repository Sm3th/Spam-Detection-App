import streamlit as st
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Sayfa ayarı
st.set_page_config(
    page_title="Spam Detector App",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="expanded"
)

# NLTK indir
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

# Veri yükle
@st.cache_data
def load_data():
    df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'text']
    return df

df = load_data()

# Temizleme fonksiyonları
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def clean_text(text):
    text = text.lower()
    allowed = "$%!"
    text = "".join([c for c in text if c not in string.punctuation or c in allowed])
    words = text.split()
    tagged_words = pos_tag(words)
    lemmatized = [
        WordNetLemmatizer().lemmatize(word, get_wordnet_pos(tag))
        for word, tag in tagged_words if word not in stopwords.words('english')
    ]
    return " ".join(lemmatized)

df['clean_text'] = df['text'].apply(clean_text)

# TF-IDF
vectorizer = TfidfVectorizer(ngram_range=(1, 4), min_df=3, max_df=0.9)
X = vectorizer.fit_transform(df['clean_text'])
y = df['label'].map({'ham': 0, 'spam': 1})

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
nb_model = MultinomialNB().fit(X_train, y_train)
log_model = LogisticRegression(max_iter=1000).fit(X_train, y_train)

# Tahmin
def predict_models(email):
    cleaned = clean_text(email)
    vec = vectorizer.transform([cleaned])
    return nb_model.predict(vec)[0], log_model.predict(vec)[0]

# Arayüz
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Spam Email Detector</h1>", unsafe_allow_html=True)
st.write("Enter an email message and the system will predict if it is Spam or Not Spam.")

email_input = st.text_area("Email content:")

if st.button("Check Spam"):
    if email_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        nb_result, log_result = predict_models(email_input)
        st.subheader("Model Predictions:")
        if nb_result == 1:
            st.error("Naive Bayes Prediction: Spam ❌")
        else:
            st.success("Naive Bayes Prediction: Not Spam ✅")
        if log_result == 1:
            st.error("Logistic Regression Prediction: Spam ❌")
        else:
            st.success("Logistic Regression Prediction: Not Spam ✅")
