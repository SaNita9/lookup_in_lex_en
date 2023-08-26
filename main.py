from nrclex import NRCLex
import numpy as np
import pandas as pd
import spacy

from data3 import data

nlp = spacy.load("en_core_web_sm")


def lexicon_search():
    no_examples = 0
    got_right = 0
    neutrals = 0
    dialogue_index = 1
    for item in data:
        dialogue = item["utterances"]
        ac_emotions = item['emotions']

        print(f"Dialogue no.{dialogue_index}")
        dialogue_index += 1
        utterance_index = 0

        for utterance in dialogue:
            print(utterance)
            utterance = nlp(utterance)
            lemmatized_tokens = []
            for token in utterance:
                lemmatized_tokens.append(
                    token.lemma_)  # in cazul in care am nevoie de o lista de tokens deja prelucrati
            lemmatized_sentence = " ".join([token for token in lemmatized_tokens])
            prediction_lists = NRCLex(lemmatized_sentence)
            if prediction_lists.affect_dict:
                emotions_with_confidence = prediction_lists.top_emotions
            else:
                emotions_with_confidence = [("neutral", 1.0)]
                neutrals += 1
            print(f"predicted emotions: {emotions_with_confidence}")
            print(f"prediction was based on: {prediction_lists.affect_dict}")
            print(f"actual emotion: {ac_emotions[utterance_index]}\n")
            top_emotions_list = []
            for em, conf in emotions_with_confidence:
                top_emotions_list.append(em)
            if ac_emotions[utterance_index] in top_emotions_list:
                got_right += 1
            utterance_index += 1
            no_examples += 1
    accuracy = got_right / no_examples * 100
    comparison = neutrals / no_examples * 100
    print(f"accuracy: {accuracy}%")
    print(f"accuracy if we just assumed every utterance was neutral: {comparison}%")


lexicon_search()
