import nltk 
import re
import string
#download required NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

text = ("I am Che Ma Kinh Lac, a twenty-year-old individual born on November 5, 2005, in the vibrant Ho Chi Minh City, Vietnam."
        "My phone number is 0902189341, and my email is kinhlacma@gmail.com"
"My personal interests involve the intricate art of crochet and the immersive worlds of video games. "
"I am deeply passionate about expanding my knowledge in linguistics, advanced technology, and the English language, striving to understand the complex intersections between human communication and modern digital innovation. " 
)
print("ORIGINAL TEXT:\n", text)
print("-" * 60)

#Tokenization
tokens = word_tokenize(text)
print("TOKENS:\n", tokens)
print("-" * 60)

#Lowercasing
lower_tokens = [token.lower() for token in tokens]
print("LOWERCASED TOKENS:\n", lower_tokens)
print("-" * 60)

#Stopword Removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in lower_tokens if token not in stop_words]
print("FILTERED TOKENS (STOPWORDS REMOVED):\n", filtered_tokens)
print("-" * 60)

#Punctuation Removal
punctuation_table = str.maketrans('', '', string.punctuation)
punctuation_free_tokens = [token.translate(punctuation_table) for token in filtered_tokens if token.strip()]
print("PUNCTUATION-FREE TOKENS:\n", punctuation_free_tokens)
print("-" * 60)

#Steming
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(token) for token in
punctuation_free_tokens if token]
print("STEMMED TOKENS:\n", stemmed_tokens)
print("-" * 60)

#Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for
token in punctuation_free_tokens if token]
print("LEMMATIZED TOKENS:\n", lemmatized_tokens)