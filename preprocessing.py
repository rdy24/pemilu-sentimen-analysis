import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# cleaning text
def cleaning_text(text):
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'^RT[\s]+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub(r'\\n', '', text)
    text = re.sub(r':', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


# case folding
def case_folding(text):
    return text.lower()


# tokenizing
def tokenizing(text):
    return word_tokenize(text)


# stopword removal
def stopword_removal(text):
    stop_words = set(stopwords.words('indonesian'))
    return [word for word in text if word not in stop_words]


# stemming
def stemming(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return [stemmer.stem(word) for word in text]
