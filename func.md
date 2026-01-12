# NLTK Library Functions Reference

## 1. Text Preprocessing Functions

### Tokenization:
- `nltk.word_tokenize(text)` - Splits text into individual words/tokens
- `nltk.sent_tokenize(text)` - Splits text into sentences
- `nltk.RegexpTokenizer(pattern)` - Tokenizes using regular expressions
- `nltk.TreebankWordTokenizer()` - Tokenizes using Penn Treebank conventions

### Stopword Removal:
- `nltk.corpus.stopwords.words('language')` - Gets list of stopwords for a language
- Example: `set(nltk.corpus.stopwords.words('english'))`

### Normalization:
- `text.lower()` - Converts to lowercase
- `string.punctuation` - Contains punctuation characters for removal

## 2. Morphological Analysis Functions

### Stemming:
- `nltk.PorterStemmer().stem(word)` - Applies Porter stemming algorithm
- `nltk.SnowballStemmer(language).stem(word)` - Applies Snowball stemming
- `nltk.LancasterStemmer().stem(word)` - Applies Lancaster stemming

### Lemmatization:
- `nltk.WordNetLemmatizer().lemmatize(word, pos)` - Lemmatizes with part-of-speech
- `nltk.corpus.wordnet.morphy(word, pos)` - Basic lemmatization function

## 3. Part-of-Speech (POS) Tagging Functions

- `nltk.pos_tag(tokens)` - Tags tokens with POS tags
- `nltk.pos_tag_sents(sentences)` - Tags multiple sentences
- `nltk.map_tag(source, target, tag)` - Maps between different tag sets
- `nltk.RegexpTagger(patterns)` - Uses regex patterns for tagging

## 4. Parsing Functions

- `nltk.RecursiveDescentParser(grammar)` - Recursive descent parser
- `nltk.ShiftReduceParser(grammar)` - Shift-reduce parser
- `nltk.ChartParser(grammar)` - Chart parsing algorithm
- `nltk.EarleyChartParser(grammar)` - Earley chart parser

## 5. Named Entity Recognition Functions

- `nltk.ne_chunk(tagged_tokens)` - Performs named entity recognition
- `nltk.ne_chunk_sents(tagged_sentences)` - Chunks multiple sentences

## 6. Sentiment Analysis Functions

- `nltk.sentiment.vader.SentimentIntensityAnalyzer()` - For social media sentiment
- `analyzer.polarity_scores(text)` - Returns positive, negative, neutral scores
- `score['compound']` - Overall compound score (-1 to 1)

## 7. Corpora Access Functions

- `nltk.corpus.brown.words()` - Access Brown Corpus
- `nltk.corpus.gutenberg.words()` - Access Project Gutenberg texts
- `nltk.corpus.reuters.words()` - Access Reuters news corpus
- `nltk.corpus.wordnet.synsets(word)` - Access WordNet synonyms
- `nltk.corpus.stopwords.words('english')` - Access stopwords

## 8. Statistical Analysis Functions

- `nltk.FreqDist(tokens)` - Frequency distribution of tokens
- `nltk.ConditionalFreqDist(pairs)` - Conditional frequency distribution
- `nltk.bigrams(tokens)` - Creates bigram sequences
- `nltk.trigrams(tokens)` - Creates trigram sequences
- `nltk.ngrams(tokens, n)` - Creates n-gram sequences
- `nltk.collocations.BigramCollocationFinder.from_words(words)` - Finds collocations

## 9. Classification Functions

- `nltk.NaiveBayesClassifier.train(training_data)` - Trains Naive Bayes classifier
- `nltk.MaxentClassifier.train(training_data)` - Trains MaxEnt classifier
- `nltk.DecisionTreeClassifier.train(training_data)` - Trains decision tree
- `nltk.classify.accuracy(classifier, test_set)` - Computes classifier accuracy

## 10. Information Extraction Functions

- `text.concordance(word)` - Shows word in context
- `text.similar(word)` - Finds similar words in context
- `text.dispersion_plot([words])` - Plots word dispersion
- `nltk.re_show(pattern, text)` - Shows regex matches in text

## 11. Language Modeling Functions

- `nltk.model.NgramModel(n, vocabulary)` - Creates n-gram model
- `probability(word | context)` - Calculates word probability given context
- `model.generate(length)` - Generates text using the model

## 12. Text Similarity Functions

- `nltk.jaccard_distance(set1, set2)` - Jaccard similarity measure
- `nltk.edit_distance(str1, str2)` - Edit distance between strings
- `nltk.metrics.distance.edit_distance(str1, str2)` - Levenshtein distance
- `nltk.metrics.scores.cosine_similarity(vec1, vec2)` - Cosine similarity

## 13. Phonetics Functions

- `nltk.soundex(word)` - Converts word to Soundex code
- `nltk.metaphone(word)` - Converts word to Metaphone code

## 14. Utility Functions

- `nltk.download(resource)` - Downloads NLTK resources
- `nltk.help()` - Provides help on NLTK objects
- `nltk.CFG.fromstring(string)` - Creates context-free grammar from string
- `nltk.probability.ProbDistI` - Abstract base class for probability distributions

## 15. Application Functions

- `nltk.chat.Chat(pairs, reflections)` - Creates chatbot
- `nltk.probability.LaplaceProbDist(freqdist)` - Laplace smoothing
- `nltk.probability.ELEProbDist(freqdist)` - Expected likelihood estimation