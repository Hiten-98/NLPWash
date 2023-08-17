from NLPWash import textscrub


a = textscrub.TextProcessor()

example = "@105841 Do you need help? Please DM us and let us know any steps you've tried since experiencing these behaviors.😭😭 https://t.co/GDrqU22YpT"
output = a.normalize_text(example,
                                lowercase=True,
                                remove_hyperlinks=False,
                                remove_emojis=True,
                                remove_html_tags = True,
                                remove_punctuation=True,
                                tokenization=True,
                                remove_stopwords=True,
                                stemming = None,
                                lemmatizing=True
                                )

print("Raw Text: " + example)
print("Clean Text: " +output)

