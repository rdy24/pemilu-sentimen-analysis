import string
import re
import emoji
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


# cleaning text
def cleaning_text(text):
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'^RT[\s]+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = emoji.demojize(text)
    text = re.sub(r':\S+:', ' ', text)
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))

    return text


# case folding
def case_folding(text):
    return text.lower()


# tokenizing
def tokenizing(text):
    return word_tokenize(text)


# stopword removal
def stopword_removal(text):
    factory = StopWordRemoverFactory()
    stopword = factory.create_stop_word_remover()
    more_stopword = [
        "ah", "u", "az", "gp", "maniaaaa", "unk", "yaa", "all", "in", "md", "waahh", "halooo", "bro", "x", "e", "ktp",
        "kwkw", "se", "adi", "wni", "n", "wew", "k", "tps", "tu", "psi", "pbb", "jauuuuuhhhh", "anka",
        "bppemilu", "sorry", "wkwkwk", "pra", "rk", "et", "bu", "nm", "wkakakaka", "wkwk", "waww", "ta", "that", "s",
        "the", "key", "si", "simbolon", "ni", "a", "jr",
        "lokalqn", "antonio", "make", "sense", "vs", "nasdem", "ps", "jkw", "z", "umkm", "ko", "an", "genz", "ganjarrr",
        "atass", "buzerrr", "ji", "fake", "jkt", "dki", "h", "ikn", "framming", "poltracking", "chuakssss", "isl",
        "pildun", "yah", "hayooo", "drunn", "wau", "ri", "wait", "and", "see", "sponsored", "by", "metrotivu",
        "manies", "mk", "warni", "kpk", "smart", "cak", "tol", "y", "b", "wkwkkw", "ip", "hp",
        "njir", "g", "buggy", "bukep", "aldi", "taher", "pkb", "pdfi", "nu", "one", "man", "dpr", "r", "t", "rt", "pks",
        "yq", "hhahahhaa", "tk", "jokowiiiii", "kamuuu", "ngapainnn", "laaah", "wiiiees", "he", "rdebaru", "tohir",
        "ektp", "hopifah", "yyess", "klw", "d", "ky", "ahy", "aje", "jan", "etes", "de", "pst", "hikhikhik", "kaésang",
        "â", "ï", "mbak", "donny", "expectation", "ðÿœ", "siiiiih", "gemesiiiiiin", "gaissssssss", "suksesssss", "plt",
        "sub", "jirrrr", "bet", "zaken", "tphm", "terkesan", "ngak", "puan", "maharani", "al", "ala", "nur", "děně",
        "pp", "dct", "muh", "berau", "icdnr", "war", "bani", "j", "voc", "thok", "apk", "law", "kh", "arjun", "khan",
        "hut", "kab", "rp", "l", "bahahahaha", "ppk", "pps", "maluku", "joel", "v", "ii", "iv", "pepabri", "dcs", "gn",
        "iii", "igor", "spin", "pal", "liah", "nih", "lu", "segalanyaaaaaa", "pemilu", "say", "lah", "lahh", "wawww",
        "wah", "wktotlah", "yak", "hehehe", "wkwkwkw", "tuhh", "siihh", "wow", "eh", "nak", "kak", "oke", "koq", "gps",
        "location", "btw", "mba", "lho", "hemm", "wlwkwkw", "wkwwk", "jwbrt", "kpu", "bawaslu", "²", "²an", "dgngolkar",
        "q", "wakìl", "i", "nkr", "at", "mf", "is", "best", "next", "right", "on", "placerisma", "xnxx", "so", "jis",
        "dp", "rb", "hahahahahahahahah", "haam", "sos", "m", "ak", "pt", "ma", "nw", "ugm", "pj", "asn", "fyi", "uns",
        "²ðÿ", "ðÿ", "âœœ", "megagiliran", "thm menteri", "tooo", "nee", "frp", "nelikung", "lskux", "guys"
    ]
    stop_words = factory.get_stop_words() + more_stopword
    return [word for word in text if word not in stop_words]


# stemming
def stemming(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return [stemmer.stem(word) for word in text if isinstance(word, str)]


# normalisasi
normalisasi_kata_df = pd.read_csv('normalisasi-new.csv')
normalisasi_kata_dict = dict(zip(normalisasi_kata_df['before'], normalisasi_kata_df['after']))

def normalisasi(text):
    return [normalisasi_kata_dict.get(word, word) for word in text]


