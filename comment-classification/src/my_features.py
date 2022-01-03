# encoding: utf-8
import pandas as pd
import string
from nltk import pos_tag
from sklearn.base import TransformerMixin
from nltk.corpus import stopwords
from collections import Counter
import unicodedata

eng_stopwords = set(stopwords.words("english"))
badwords = pd.read_csv('../data/bad-words.csv', header=None).iloc[:,0].tolist()

# Parts of Speech Tag Count
class PoS_TagFeatures(TransformerMixin):
    def tag_PoS(self, text):
        text_splited = text.split(' ')
        text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
        text_splited = [s for s in text_splited if s]
        pos_list = pos_tag(text_splited)
        noun_count = len([w for w in pos_list if w[1] in ('NN', 'NNP', 'NNPS', 'NNS')]) + 0.01
        adjective_count = len([w for w in pos_list if w[1] in ('JJ', 'JJR', 'JJS')]) + 0.01
        verb_count = len([w for w in pos_list if w[1] in ('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ')]) + 0.01

        words = len(text.split()) + 0.01
        length = len(text) + 0.01
        return [noun_count, noun_count / words, noun_count / length,
                adjective_count, adjective_count / words, adjective_count / length,
                verb_count, verb_count / words, verb_count / length]

    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        return [{'nouns': counts[0],
                 'nounNormByWords': counts[1],
                 'nounNormByLength': counts[2],
                 'adjectives': counts[3],
                 'adjectiveNormByWords': counts[4],
                 'adjectiveNormByLength': counts[5],
                 'verbs': counts[6],
                 'verbNormByWords': counts[7],
                 'verbNormByLength': counts[8]}
                for counts in map(self.tag_PoS, posts)]

# Bad Words Occurrence Count
class BadWords_Features(TransformerMixin):
    def badWordCount(self, text):
        badCount = sum(text.count(w) for w in badwords) + 0.01
        return [badCount, badCount / (len(text.split()) + 0.01), badCount / (len(text) + 0.01)]

    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        return [{'badwordcount': badCounts[0],
                 'badNormByWords': badCounts[1],
                 'badNormByLength': badCounts[2]}
                for badCounts in map(self.badWordCount, posts)]


# Symbol Occurrence Count
class Symbol_Features(TransformerMixin):
    def symbolCount(self, text):
        foul_filler = sum(text.count(w) for w in '*&$%@#!') + 0.01
        userMentions = text.count("User:") + 0.01
        smileys = sum(text.count(w) for w in (':-)', ':)', ';-)', ';)')) + 0.01
        exclamation = text.count("!") + 0.01
        question = text.count("User:") + 0.01
        punctuation = sum(text.count(w) for w in string.punctuation) + 0.01
        all_symbol = sum(text.count(w) for w in unicodedata.normalize('NFKD',u'*&#$%“”¨«»®´·º½¾¿¡§£₤‘’').encode('ascii','ignore')) + 0.01


        words = len(text.split()) + 0.01
        length = len(text) + 0.01
        return [foul_filler, foul_filler / words, foul_filler / length,
                userMentions, userMentions / words, userMentions / length,
                smileys, smileys / words, smileys / length,
                exclamation, exclamation / words, exclamation / length,
                question, question / words, question / length,
                punctuation, punctuation / words, punctuation / length,
                all_symbol, all_symbol / words, all_symbol / length]

    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        return [{'foul_fillerCount': symCounts[0],
                 'foul_fillerNormByWords': symCounts[1],
                 'foul_fillerNormByLength': symCounts[2],
                 'userMentionsCount': symCounts[3],
                 'userMentionsNormByWords': symCounts[4],
                 'userMentionsNormByLength': symCounts[5],
                 'smileysCount': symCounts[6],
                 'smileysNormByWords': symCounts[7],
                 'smileysNormByLength': symCounts[8],
                 'exclamationCount': symCounts[9],
                 'exclamationNormByWords': symCounts[10],
                 'exclamationNormByLength': symCounts[11],
                 'questionCount': symCounts[12],
                 'questionNormByWords': symCounts[13],
                 'questionNormByLength': symCounts[14],
                 'punctuationCount': symCounts[15],
                 'punctuationNormByWords': symCounts[16],
                 'punctuationNormByLength': symCounts[17],
                 'all_symbolCount': symCounts[18],
                 'all_symbolNormByWords': symCounts[19],
                 'all_symbolNormByLength': symCounts[20]}
                for symCounts in map(self.symbolCount, posts)]


# General Text Based Features
class TextFeatures(TransformerMixin):
    def featureCount(self, text):
        words = len(text.split()) + 0.01
        length = len(text) + 0.01
        capitals = sum(1 for c in text if c.isupper()) + 0.01
        paragraphs = text.count('\n') + 0.01
        stopwords = sum(text.count(w) for w in eng_stopwords) + 0.01
        unique = len(set(w for w in text.split())) + 0.01
        word_counts = Counter(text.split())
        repeat = sum(count for word, count in sorted(word_counts.items()) if count > 10) + 0.01

        return [words, length, words / length,
                capitals, capitals / words, capitals / length,
                paragraphs, paragraphs / words, paragraphs / length,
                stopwords, stopwords / words, stopwords / length,
                unique, unique / words, unique / length,
                repeat, repeat / words, repeat / length]

    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        return [{'words': counts[0],
                 'length': counts[1],
                 'wordNormByLength': counts[2],
                 'capitalsCount': counts[3],
                 'capitalsNormByWords': counts[4],
                 'capitalsNormByLength': counts[5],
                 'paragraphsCount': counts[6],
                 'paragraphsNormByWords': counts[7],
                 'paragraphsNormByLength': counts[8],
                 'stopwordsCount': counts[9],
                 'stopwordsNormByWords': counts[10],
                 'stopwordsNormByLength': counts[11],
                 'uniqueCount': counts[12],
                 'uniqueNormByWords': counts[13],
                 'uniqueNormByLength': counts[14],
                 'repeatCount': counts[15],
                 'repeatNormByWords': counts[16],
                 'repeatNormByLength': counts[17]}
                for counts in map(self.featureCount, posts)]
