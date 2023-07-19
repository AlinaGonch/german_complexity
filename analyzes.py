import os
import json
import statistics
from collections import Counter, defaultdict
import math
import random
import gzip
import spacy
import nltk

from nltk import tokenize
from math import log2
from preproccess_model import load_data

nltk.download('punkt')
# count type/token ratio
def ttr_with_window(tokens):
    window_size = 13000

# Calculate TTR with a sliding window
    ttr_values = []
    for i in range(len(tokens) - window_size + 1):
        window = tokens[i:i + window_size]
        types = set(window)
        ttr = len(types) / window_size
        ttr_values.append(ttr)

    # Print the TTR values
    
    print('TTR_sd', statistics.stdev(ttr_values))
    print("TTR value:", sum(ttr_values)/len(ttr_values))

# mean size of paradigm
def mps(tokens, lemmas):
    window_size = 13000
    msp_values = []
    for i in range(len(tokens) - window_size + 1):
        window_tokens = tokens[i:i + window_size]
        window_lemmas = lemmas[i:i + window_size]
        word_form_types = len(set(window_tokens))
        lemma_types = len(set(window_lemmas))
        msp_values.append(word_form_types / lemma_types)
    

    print("MSP value:", sum(msp_values)/len(msp_values))
    print('MSP_SD:', statistics.stdev(msp_values))

# count lemma entropy
def lh(lemmas):
    window_size = 13000
    lemmas_ent_values = []
    for i in range(len(lemmas) - window_size + 1):
        window_lemmas = lemmas[i:i + window_size]
        lemma_frequencies = Counter(window_lemmas)
        total_lemmas = len(window_lemmas)
        lemma_probabilities = {lemma: freq / total_lemmas for lemma, freq in lemma_frequencies.items()}
        lemma_entropy = -sum(prob * math.log2(prob) for prob in lemma_probabilities.values())
        lemmas_ent_values.append(lemma_entropy)
    
    print('Lemmas entropy:', sum(lemmas_ent_values)/ len(lemmas_ent_values))
    print('LH_SD:', statistics.stdev(lemmas_ent_values))

# count word entropy
def wh(tokens):
    window_size = 13000
    words_values = []
    for i in range(len(tokens) - window_size + 1):
        wtokens = tokens[i:i + window_size]
        word_frequencies = Counter(wtokens)

        # Total number of words
        total_words = len(wtokens)

        # Calculate word probabilities
        word_probabilities = {word: freq / total_words for word, freq in word_frequencies.items()}

        # Calculate word entropy
        word_entropy = -sum(prob * math.log2(prob) for prob in word_probabilities.values()) 
        words_values.append(word_entropy)
    
    print('WH_value:', sum(words_values)/ len(words_values))
    print('WH_SD:', statistics.stdev(words_values))
     

# count Inflectional Synthesis
def is_v(words_annot):
    window_size = 680
    is_scores = []

    for i in range(len(words_annot.keys()) - window_size + 1):
        verb_inflectional_features = defaultdict(set)
        sample =  dict(list(words_annot.items())[i:i + window_size])
        for lemma, features in sample.items():
            for feature in features:
                new_feature = ''
                for el in feature:
                    feature_name, feature_value = el.split('=')
                    new_feature += feature_value
                verb_inflectional_features[lemma].add(new_feature)

        max_inflectional_features = max(len(features) for features in verb_inflectional_features.values())


        # Calculate the index of synthesis (IS) for verb morphology
        is_scores.append(max_inflectional_features)
    
    print('IS_value:', sum(is_scores)/ len(is_scores))
    print('IS_SD:', statistics.stdev(is_scores))

    
# convert features into the dictionary format for the futher analyzes
def convert_data(data):
    converted_data = []
    for entry in data:
        word = entry[0]
        features = {}
        for feature in entry[1]:
            feature_name, feature_value = feature.split('=')
            if feature_name != feature_value:
                if feature_name != 'pos':
                    features[feature_name] = feature_value
        converted_entry = { **features}
        converted_data.append(converted_entry)
    return converted_data

 # count feature -value entropy   
def mfh(data):
    window_size = 13000
    feature_entropies = []
    feature_value_dict = convert_data(data)
    # Convert the list of feature-value pairs to a dictionary 
    for i in range(len(data) - window_size + 1):
    # Count the frequency of each feature-value pair
        feature_value_frequencies = Counter((fv for sentence in feature_value_dict[i:i + window_size] for fv in sentence.items()))

        # Total number of feature-value pairs
        total_pairs = len(feature_value_dict[i:i + window_size])

        # Calculate the probability of each feature-value pair
        feature_value_probabilities = {fv: freq / total_pairs for fv, freq in feature_value_frequencies.items()}

        # Calculate the morphological feature entropy (MFH)
        feature_entropy = -sum(prob * math.log2(prob) for prob in feature_value_probabilities.values())
        feature_entropies.append(feature_entropy)

    print('MFH_value:', sum(feature_entropies)/ len(feature_entropies))
    print('MFH_SD:', statistics.stdev(feature_entropies))
import random

# calculate the difference between entropies of original text and distored text
def calculate_morphological_complexity(original_text, distorted_text):
    # Calculate the entropy of the original text
    original_entropy = calculate_entropy(original_text)

    # Calculate the entropy of the distorted text
    distorted_entropy = calculate_entropy(distorted_text)

    # Calculate the difference in entropy between the original and distorted texts
    entropy_difference = original_entropy - distorted_entropy

    return entropy_difference

def calculate_entropy(text):
    # Count the frequency of each character in the text
    char_frequency = {}
    total_chars = 0
    for char in text:
        if char not in char_frequency:
            char_frequency[char] = 0
        char_frequency[char] += 1
        total_chars += 1

    # Calculate the probability of each character
    char_probabilities = {char: count / total_chars for char, count in char_frequency.items()}

    # Calculate the entropy
    entropy = -sum(prob * log2(prob) for prob in char_probabilities.values())

    return entropy


def distort_text(text):
    # Split the text into words
    words = text.split()

    # Distort each word
    distorted_words = []
    for word in words:
        # Distort the word by shuffling its characters
        distorted_word = ''.join(random.sample(word, len(word)))
        distorted_words.append(distorted_word)

    # Join the distorted words back into a single text
    distorted_text = ' '.join(distorted_words)

    return distorted_text


# count Information in Word Structure
def ws(text):
    text_joined = tokenize.sent_tokenize(text)
    ws_info = []
    for original_text in text_joined:
        distorted_text = distort_text(original_text)
        ws_info.append(calculate_morphological_complexity(original_text, distorted_text))
    print("Word Structure Information (WS):", sum(ws_info)/ len(ws_info))
    print('WS_SD:', statistics.stdev(ws_info))


#select verbs from the data
def verbs_selection(lemma, data):
    verbs= {}
    for i, word in enumerate(data):
        if 'pos=V' in word[1]:
            if lemma[i] in verbs.keys():
                verbs[lemma[i]].append(word[1])
            else:
                verbs[lemma[i]]= [word[1]]
    for key, value in verbs.items():
        verbs[key] = [list(x) for x in set(tuple(lst) for lst in value)]
    return verbs


        


if __name__ == '__main__':
    old_data_full = json.load(open('1840_tokken_lemmas.json', 'r', encoding='utf-8'))
    modern_data_full = json.load(open('1924_token_lemmas.json', 'r', encoding='utf-8'))
    annotated_old_data = json.load(open('1840_annotated_data.json', 'r', encoding='utf-8'))
    annotated_modern_data = json.load(open('1924_annotated_data.json', 'r', encoding='utf-8'))
    annotated_old_lemmas = json.load(open('1840_lemmas_annotated_data.json', 'r', encoding='utf-8'))
    annotated_modern_lemmas = json.load(open('1924_lemmas_annotated_data.json', 'r', encoding='utf-8'))
   
    texts = load_data('15_ex')
    modern_text = texts[1]
    old_text = texts[0]

    old_tokens = [keys[0] for keys in old_data_full]
    modern_tokens = [keys[0] for keys in modern_data_full]

    old_lemmas = [keys[1] for keys in old_data_full]
    modern_lemmas = [keys[1] for keys in modern_data_full]

    print(len(annotated_modern_data), len(annotated_modern_lemmas))
    print(len(annotated_old_data), len(annotated_old_lemmas))

    old_verbs = verbs_selection(annotated_old_lemmas, annotated_old_data)
    modern_verbs = verbs_selection(annotated_modern_lemmas, annotated_modern_data)

    nlp = spacy.load('de_core_news_md', disable=['tagger'])
    nlp.max_length = 10000000

    old_doc = nlp(old_text)
    modern_doc = nlp(modern_text)

    print('Old corpus data:')
    ttr_with_window(old_tokens)
    mps(old_tokens, old_lemmas)
    wh(old_tokens)
    lh(old_lemmas)
    mfh(annotated_old_data)
    is_v(old_verbs)
    ws(old_text)
    print('_______________')
    
    print('Modern corpus data:')
    ttr_with_window(modern_tokens)
    mps(modern_tokens, modern_lemmas)  
    wh(modern_tokens)
    lh(modern_lemmas)
    mfh(annotated_modern_data)
    is_v(modern_verbs)
    ws(modern_text)


