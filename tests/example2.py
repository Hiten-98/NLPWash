import pandas as pd
from NLPWash import textscrub

df = pd.read_csv(r'D:\NMIMS-Hiten\NLP\Disaster Tweets\train.csv')
df.head()

a = textscrub.TextProcessor()

df['normalized_text'] = df['text'].apply(a.normalize_text,
                                        lowercase=True,
                                        tokenization=True,
                                        remove_stopwords=True,
                                        remove_emojis=True,
                                        remove_html_tags = True,
                                        remove_punctuation=True,
                                        remove_hyperlinks= True,
                                        stemming = 'SnowballStemmer',
                                        lemmatizing=True)
df.head()
