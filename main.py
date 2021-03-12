#Required Libraries
from spacy.lang.en.stop_words import STOP_WORDS
from itertools import product
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
from gensim.models import Word2Vec, KeyedVectors   
from nltk.corpus import stopwords
from nltk.stem.porter import *
from gensim.models import word2vec
from urllib.request import urlopen
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import CountVectorizer
from collections import OrderedDict
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.tokenize import word_tokenize
from nltk import tokenize
from operator import itemgetter
import pandas as pd
import numpy as np
import heapq
import nltk
import os
import re
import spacy
import datetime
import gensim
import math


nltk.download('stopwords')
nltk.download('wordnet')
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer() 
stops = set(stopwords.words("english"))
nlp = spacy.load('en_core_web_lg')



class Content_Extract:
    def __init__(self, url):
        self.url=url
    
    def content_extract(self,url):
        def condense_newline(text):
            return '\n'.join([p for p in re.split('\n|\r', text) if len(p) > 0])
        page = urlopen(self.url)
        html = page.read().decode("utf-8")
        soup = BeautifulSoup(html, 'html.parser')
        TAGS = ['p','h1','h2','h3','h4','h5','h6','h7']
        content=' '.join([condense_newline(tag.text) for tag in soup.findAll(TAGS)])
        
        return content
		


class WordEmbedding: 

    def __init__(self, verbose=0):
        self.verbose = verbose
        self.model = {}
        
    def convert(self, source, ipnut_file_path, output_file_path):
        if source == 'glove':
            input_file = datapath(ipnut_file_path)
            output_file = get_tmpfile(output_file_path)
            glove2word2vec(input_file, output_file)
        elif source == 'word2vec':
            pass
        elif source == 'fasttext':
            pass
        elif source == 'homemade_embedding':
            pass
        else:
            raise ValueError('Possible value of source are glove, word2vec, fasttext, or HomeMadeEmbedding')
        
    def load(self, source, file_path):
        print(datetime.datetime.now(), 'start: loading', source)
        if source == 'glove':
            self.model[source] = gensim.models.KeyedVectors.load_word2vec_format(file_path)
        elif source == 'word2vec':
            self.model[source] = gensim.models.KeyedVectors.load_word2vec_format(file_path, binary=True)
        elif source == 'fasttext':
            self.model[source] = gensim.models.wrappers.FastText.load_fasttext_format(file_path)
        elif source == 'homemade_embedding':
            self.model[source] = gensim.models.KeyedVectors.load_word2vec_format(file_path, binary=True)
        else:
            raise ValueError('Possible value of source are glove, word2vec, fasttext, or HomeMadeEmbedding')
            
        print(datetime.datetime.now(), 'end: loading', source)
            
        return self
    
    def get_model(self, source):
        if source not in ['glove', 'word2vec', 'fasttext', 'homemade_embedding']:
            raise ValueError('Possible value of source are glove, word2vec, fasttext, or HomeMadeEmbedding')
            
        return self.model[source]
    
    def get_words(self, source, size=None):
        if source not in ['glove', 'word2vec', 'fasttext', 'homemade_embedding']:
            raise ValueError('Possible value of source are glove, word2vec, fasttext, or HomeMadeEmbedding')
        
        if source in ['glove', 'word2vec','homemade_embedding']:
            if size is None:
                return [w for w in self.get_model(source=source).vocab]
            elif size is None:
                return [w for w in self.get_model(source=source).vocab]
            else:
                results = []
                for i, word in enumerate(self.get_model(source=source).vocab):
                    if i >= size:
                        break
                        
                    results.append(word)
                return results
            
        elif source in ['fasttext']:
            if size is None:
                return [w for w in self.get_model(source=source).wv.vocab]
            else:
                results = []
                for i, word in enumerate(self.get_model(source=source).wv.vocab):
                    if i >= size:
                        break
                        
                    results.append(word)
                return results
        
        return Exception('Unexpected flow')
    
    def get_dimension(self, source):
        if source not in ['glove', 'word2vec', 'fasttext', 'homemade_embedding']:
            raise ValueError('Possible value of source are glove, word2vec, fasttext, or HomeMadeEmbedding')
        
        if source in ['glove', 'word2vec','homemade_embedding']:
            return self.get_model(source=source).vectors[0].shape[0]
            
        elif source in ['fasttext']:
            word = self.get_words(source=source, size=1)[0]
            return self.get_model(source=source).wv[word].shape[0]
        
        return Exception('Unexpected flow')
    
    def get_vectors(self, source, words=None):
        if source not in ['glove', 'word2vec', 'fasttext', 'homemade_embedding']:
            raise ValueError('Possible value of source are glove, word2vec, fasttext, or HomeMadeEmbedding')
        
        if source in ['glove', 'word2vec', 'fasttext', 'homemade_embedding']:
            if words is None:
                words = self.get_words(source=source)
            
            embedding = np.empty((len(words), self.get_dimension(source=source)), dtype=np.float32)            
            for i, word in enumerate(words):
                embedding[i] = self.get_vector(source=source, word=word)
                
            return embedding
        
        return Exception('Unexpected flow')
    
    def get_vector(self, source, word, oov=None):
        if source not in ['glove', 'word2vec', 'fasttext', 'homemade_embedding']:
            raise ValueError('Possible value of source are glove, word2vec, fasttext, or HomeMadeEmbedding')
            
        if source not in self.model:
            raise ValueError('Did not load %s model yet' % source)
        
        try:
            return self.model[source][word]
        except KeyError as e:
            raise
            
    def get_synonym(self, source, word, oov=None):
        if source not in ['glove', 'word2vec', 'fasttext', 'homemade_embedding']:
            raise ValueError('Possible value of source are glove, word2vec, fasttext, or HomeMadeEmbedding')
            
        if source not in self.model:
            raise ValueError('Did not load %s model yet' % source)
        
        try:
            return self.model[source].most_similar(positive=word, topn=5)
        except KeyError as e:
            raise
    
    def which_distance_between_two_words(self, source, word1, word2, oov=None):
        if source not in ['glove', 'word2vec', 'fasttext', 'homemade_embedding']:
            raise ValueError('Possible value of source are glove, word2vec, fasttext, or HomeMadeEmbedding')
            
        if source not in self.model:
            raise ValueError('Did not load %s model yet' % source)
        
        try:
            return self.model[source].similarity(word1, word2)
        except KeyError as e:
            raise



#Class function to calculate TF-IDF Score
class tfidf:
    def __init__(self,content):
        self.content=content
        
        
    def tfidf_score(self,content):
        total_words = self.content.split(" ")
        total_word_length = len(total_words)
        total_sentences = tokenize.sent_tokenize(self.content)
        total_sent_len = len(total_sentences)
        tf_score = {}
        stops = set(stopwords.words("english"))
        for each_word in total_words:
            each_word = each_word.replace('.','')
            if each_word not in stops:
                if each_word in tf_score:
                    tf_score[each_word] += 1
                else:
                    tf_score[each_word] = 1
                    
        tf_score.update((x, y/int(total_word_length)) for x, y in tf_score.items())
        
        def check_sent(word, sentences): 
            final = [all([w in x for w in word]) for x in sentences] 
            sent_len = [sentences[i] for i in range(0, len(final)) if final[i]]
            return int(len(sent_len))
        
        idf_score = {}
        
        stops = set(stopwords.words("english"))
        
        for each_word in total_words:
            each_word = each_word.replace('.','')
            if each_word not in stops:
                if each_word in idf_score:
                    idf_score[each_word] = check_sent(each_word, total_sentences)
                else:
                    idf_score[each_word] = 1

        idf_score.update((x, math.log(int(total_sent_len)/y)) for x, y in idf_score.items())
        
        tf_idf_score = {key: tf_score[key] * idf_score.get(key, 0) for key in tf_score.keys()}
        
        result = dict(sorted(tf_idf_score.items(), key = itemgetter(1), reverse = True)) 
        
        return result


class TextRank4Keyword:
    """Extract keywords from text"""
    
    def __init__(self):
        self.d = 0.85 # damping coefficient, usually is .85
        self.min_diff = 1e-5 # convergence threshold
        self.steps = 10 # iteration steps
        self.node_weight = None # save keywords and its weight

    
    def set_stopwords(self, stopwords):  
        """Set stop words"""
        for word in STOP_WORDS.union(set(stopwords)):
            lexeme = nlp.vocab[word]
            lexeme.is_stop = True
    
    def sentence_segment(self, doc, candidate_pos, lower):
        """Store those words only in cadidate_pos"""
        sentences = []
        for sent in doc.sents:
            selected_words = []
            for token in sent:
                # Store words only with cadidate POS tag
                if token.pos_ in candidate_pos and token.is_stop is False:
                    if lower is True:
                        selected_words.append(token.text.lower())
                    else:
                        selected_words.append(token.text)
            sentences.append(selected_words)
        return sentences
        
    def get_vocab(self, sentences):
        """Get all tokens"""
        vocab = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1
        return vocab
    
    def get_token_pairs(self, window_size, sentences):
        """Build token_pairs from windows in sentences"""
        token_pairs = list()
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i+1, i+window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in token_pairs:
                        token_pairs.append(pair)
        return token_pairs
        
    def symmetrize(self, a):
        return a + a.T - np.diag(a.diagonal())
    
    def get_matrix(self, vocab, token_pairs):
        """Get normalized matrix"""
        # Build matrix
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype='float')
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            g[i][j] = 1
            
        # Get Symmeric matrix
        g = self.symmetrize(g)
        
        # Normalize matrix by column
        norm = np.sum(g, axis=0)
        g_norm = np.divide(g, norm, where=norm!=0) # this is ignore the 0 element in norm
        
        return g_norm

    
    def get_keywords(self, number=10):
        """Print top number keywords"""
        ret=list()
        node_weight = OrderedDict(sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True))
        for i, (key, value) in enumerate(node_weight.items()):
            ret.append(key)
            if i > number:
                break
        return ret
        
        
    def analyze(self, text, 
                candidate_pos=['NOUN', 'PROPN'], 
                window_size=4, lower=False, stopwords=list()):
        """Main function to analyze text"""
        
        # Set stop words
        self.set_stopwords(stopwords)
        
        # Pare text by spaCy
        doc = nlp(text)
        
        # Filter sentences
        sentences = self.sentence_segment(doc, candidate_pos, lower) # list of list of words
        
        # Build vocabulary
        vocab = self.get_vocab(sentences)
        
        # Get token_pairs from windows
        token_pairs = self.get_token_pairs(window_size, sentences)
        
        # Get normalized matrix
        g = self.get_matrix(vocab, token_pairs)
        
        # Initionlization for weight(pagerank value)
        pr = np.array([1] * len(vocab))
        
        # Iteration
        previous_pr = 0
        for epoch in range(self.steps):
            pr = (1-self.d) + self.d * np.dot(g, pr)
            if abs(previous_pr - sum(pr))  < self.min_diff:
                break
            else:
                previous_pr = sum(pr)

        # Get weight for each node
        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pr[index]
        
        self.node_weight = node_weight


#Helper Functions
def preprocess_spacy(text):
    text=re.sub("","",text)
    text=re.sub("[^a-zA-Z.]", " ", text)
    return text

def get_stop_words(stop_file_path):
    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
    return frozenset(stop_set)

def pre_process(text):
    # lowercase
    text=text.lower()
    #remove tags
    text=re.sub("","",text)
    text=text.split()
    stopwords=get_stop_words("stopwords.txt")
    #text = [stemmer.stem(word) for word in text]
    text = [lemmatizer.lemmatize(word) for word in text]
    text=[w for w in text if not w in stopwords]
    text= " ".join( text )
    # remove special characters and digits
    text=re.sub("[^a-zA-Z.]", " ", text)
    return text

def extract():
    url=input('Please mention the url : ')
    content=Content_Extract(url)
    content=content.content_extract(content)
    return content

def initital_words_extract():
    product=input('Enter product: ')
    service=input('Enter Service: ')
    product=product.lower()
    service=service.lower()
    initial_words=product.split(" ")
    for i in service.split():
        if i not in initial_words:
            initial_words.append(i)
    return initial_words
	

if __name__ == "__main__": 
	content=extract()
	#Loading word embedding files(Takes time can choose either of these 3 depending upon the use-case)
	glove_file_path = 'glove.840B.300d.vec'
	word2vec_file_path = 'GoogleNews-vectors-negative300.bin'
	fasttext_file_path = 'wiki.en.bin'  # Uncomment for better results

	word_embedding = WordEmbedding()
	word_embedding.load(source='word2vec', file_path=word2vec_file_path)
	word_embedding.load(source='glove', file_path=glove_file_path)
	#word_embedding.load(source='fasttext', file_path=fasttext_file_path) # Uncomment for better results
	text=pre_process(content)
	initial_words=initital_words_extract()
	related=dict() # A common Dictionary
	
	#Using Simialrity with all the text
	for i,j in product(initial_words,text.split(" ")):
		#print(i+" "+j)
		for source in ['glove','word2vec', 'fasttext']:
			#print('Source: %s' % (source))
			try:
				score=word_embedding.which_distance_between_two_words(source=source,word1=i, word2=j)
				if j not in related.keys():
					related.update({j:score})
				else:
					if related[j]< score:
						related.update({j:score})
			except:
				pass
	
	#Using TFIDF Scores
	t=tfidf(text)
	potential_list=t.tfidf_score(text)
	potential_list=potential_list.keys()
	for i,j in product(initial_words,potential_list):
		#print(i+" "+j)
		for source in ['glove','word2vec', 'fasttext']:
			#print('Source: %s' % (source))
			try:
				score=word_embedding.which_distance_between_two_words(source=source,word1=i, word2=j)
				if j not in related.keys():
					related.update({j:score})
				else:
					if related[j]< score:
						related.update({j:score})
			except:
				pass
	
	##Using Page Rank Algo to find most relevant words
	
	content=preprocess_spacy(content)
	length=len(content)
	tr4w = TextRank4Keyword()
	tr4w.analyze(content, candidate_pos = ['NOUN'], window_size=10, lower=True)
	l=tr4w.get_keywords(length)
	stopwords=get_stop_words("stopwords.txt")
	doc1 = [w for w in l if not w in stopwords]
	
	w1 = initial_words
	for i in w1:
		for j in l:
			tokens=nlp(i+str(" ")+j)
			token1, token2=tokens[0],tokens[1]
			if j not in related.keys():
				related.update({j:token1.similarity(token2)})
			else:
				if related[j]< score:
					related.update({j:token1.similarity(token2)})
    
	print("Related Words")
	for i in related.keys():
		if related[i]>0.50:
			print(i)