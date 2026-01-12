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

#Stopword Remova

