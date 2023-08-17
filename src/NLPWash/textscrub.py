import pandas as pd
import numpy as np
import nltk
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from string import punctuation

class TextProcessor:
    def __init__(self):
        nltk.download('punkt')  # Download NLTK data for tokenization
        nltk.download('stopwords')  # Download NLTK data for stopwords
        self.stop_words = set(stopwords.words('english'))
        nltk.download('wordnet')
        nltk.download('omw-1.4')

    def normalize_text(self, text, lowercase, remove_hyperlinks, remove_emojis,
                       remove_html_tags, remove_punctuation, tokenization, remove_stopwords, lemmatizing,
                       stemming=None):

        if lowercase:
            text = self._lowercase(text)

        if remove_emojis:
            text = self._remove_emojis(text)

        if remove_html_tags:
            text = self._remove_html_tags(text)

        if remove_punctuation:
            text = self._remove_punctuation(text)

        if remove_hyperlinks:
            text = self._remove_hyperlinks(text)

        if tokenization:
            text = self._tokenization(text)

        if remove_stopwords:
            text = self._remove_stopwords(text)

        if stemming:
            text = self._stemming(text, stemming)

        if lemmatizing:
            text = self._lemmatizing(text)

        return text

    def _lowercase(self, text):
        """
            For normalization: Converts text into lower case.
        """
        return text.lower()

    def _tokenization(self, text):
        """
            Splits texts into tokens.
        """
        return nltk.word_tokenize(text)

    def _remove_stopwords(self, text):
        """
            Removes all stopwords. It uses data of nltk stopwords.
        """
        filtered_text = [word for word in text if word.lower() not in self.stop_words]
        cleaned_text = ' '.join(filtered_text)
        return cleaned_text

    def _remove_emojis(self, text):
        """
            Removes all kind of emojis.
        """
        emoji_pattern = re.compile("["
                                   "\U0001F600-\U0001F64F"  # Emojis in U+1F600 to U+1F64F
                                   "\U0001F300-\U0001F5FF"  # Miscellaneous Symbols and Pictographs
                                   "\U0001F680-\U0001F6FF"  # Transport and Map Symbols
                                   "\U0001F700-\U0001F77F"  # Alchemical Symbols
                                   "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                                   "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                                   "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                                   "\U0001FA00-\U0001FA6F"  # Symbols and Pictographs Extended-A
                                   "\U00002702-\U000027B0"  # Dingbats
                                   "\U000024C2-\U0001F251"  # Enclosed Characters (including some emojis)
                                   "]", flags=re.UNICODE)

        clean_sentence = emoji_pattern.sub('', text)

        return clean_sentence

    def _remove_html_tags(self, text):
        """
            Removes HTML Tags including <>.
        """
        html = re.compile(r'<.*?>')
        text = re.sub(html, '', text)
        return text

    def _remove_punctuation(self, text):
        """
            Removes all sorts of punctuation.
        """
        text = re.sub('[%s]' % re.escape(punctuation), '', text)

        return text

    def _remove_hyperlinks(self, text):
        """
            Removes all hyperlinks and keeps raw texts.
        """
        text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|' \
                      '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        return text

    def _stemming(self, text, stemmer):
        """
            Normalizes words into its base form or root forms.
            It has 3 types of stemmer:
                1. SnowballStemmer
                2. LancasterStemmer
                3. PorterStemmer
            Along with the text you need to pass which stemmer you need to use.
        """
        if stemmer == 'SnowballStemmer':
            st = SnowballStemmer('english')

        elif stemmer == 'LancasterStemmer':
            st = LancasterStemmer()

        elif stemmer == 'PorterStemmer':
            st = PorterStemmer()
        words = self._tokenization(text)
        preprocessed_words = []
        for word in words:
            if word.lower() not in self.stop_words:
                preprocessed_word = st.stem(word)
                preprocessed_words.append(preprocessed_word)
        preprocessed_text = ' '.join(preprocessed_words)
        return preprocessed_text

    def _lemmatizing(self, text):
        """
            Groups together different words with the same meaning.
        """
        lemmatizer = WordNetLemmatizer()

        tokens = word_tokenize(text)
        new_text = ""

        for t in tokens:
            new_text = new_text + lemmatizer.lemmatize(t) + " "

        return new_text