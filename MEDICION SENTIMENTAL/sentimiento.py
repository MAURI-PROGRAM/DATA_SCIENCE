from textblob import TextBlob
texto='Lo mejor esta por venir'
objeto=TextBlob(text)
sentimento=objeto.sentiment.polarity
print(sentimento)