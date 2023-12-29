# import spacy
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
from nltk import pprint

# load spacy model
nlp = spacy.load("en_core_web_sm")

# load data
sentence = "Apple is looking at buying U.K. startup for $1 billion"
doc = nlp(sentence)

# print entities
pprint([(ent.text, ent.label_) for ent in doc.ents])
