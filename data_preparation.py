# datasets

import json
import re
import numpy as np
import nltk
import pandas as pd

wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')

raw_file = pd.read_csv("/Users/ignaciopalma/PycharmProjects/final_presentation/dataset/raw-health-data.csv", usecols=["phrase", "prompt"])

all_intents = [
    {
        "text": row[1]['phrase'],
        "intent": row[1]['prompt']
    }
    for row in raw_file.iterrows()
]
  
def basic_text_cleaning(text):
    text = re.sub('\s+', ' ', text)
    text = text.lower()
    text = text.strip() # leading and trailing characters removed
    text = re.sub("\'", "", text)
    text = re.sub(r'[^\w\s]', '', text) # remove punctuation
    tokens = wpt.tokenize(text)
    tokens = [token for token in tokens if not '_' in token]
    tokens = [token for token in tokens if not token.isdigit()] # remove numbers
    filtered_tokens = [token for token in tokens if token not in stop_words]
    doc = ' '.join(filtered_tokens)
    return doc

def clean_intents_array(array):
    new_array = []
    for data_dict in array:
        new_array.append({
            "text": basic_text_cleaning(data_dict['text']),
            "intent": data_dict['intent']
        })
    return new_array

def normalize_document(doc):
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
    # tokenize document
    tokens = wpt.tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc

cleaned_intents = clean_intents_array(all_intents)

# for index in range(0, len(cleaned_intents) - 1):
#   print("Original: {}".format(all_intents[index]['text']))
#   print("CLEAN: {}".format(cleaned_intents[index]['text']))
#   print("INTENT: {}".format(cleaned_intents[index]['intent']))

with open('/Users/ignaciopalma/PycharmProjects/final_presentation/dataset/clean-health-data.json', 'w') as outfile:
    json.dump(cleaned_intents, outfile)
