#使用Word2Vec

from sklearn.datasets import fetch_20newsgroups
from bs4 import BeautifulSoup
import nltk
import re
from gensim.models import word2vec

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

news=fetch_20newsgroups(subset='all')
X,y=news.data,news.target


#定义一个函数名为news_to_sentences讲每条新闻中心的句子逐一剥离出来,并返回一个句子的列表

def news_to_sentences(news):
    news_text=BeautifulSoup(news).get_text()
    tokenizer=nltk.data.load("tokenizer/punkt/english.pickle")
    raw_sentences=tokenizer.tokenize(news_text)
    sentences=[]
    for sent in raw_sentences:
        sentences.append(re.sub('[^a-zA-Z]','',sent.lower().strip()).split())
    return sentences
sentences=[]

#将长篇新闻文本中的句子玻璃出来,用于训练
for x in X:
    sentences+=news_to_sentences(x)

#配置词向量的维度
num_features=30
#保证被考虑的词汇的频度
min_word_count=20
#设定并行化训练使用CPU计算核心的数量,多核可用
num_workers=2
#定义训练词向量的上下文窗口大小
context=5
downsampling=1e-3

#训练词向量模型
model=word2vec.Word2Vec(sentences,workers=num_workers,size=num_features,min_word_count=min_word_count,window=context,sample=downsampling)

#这个设定代表当前训练好的词向量为最终版,也可以加快模型的训练速度
model.init_sims(replace=True)
print(model.most_similar("morning"))
print(model.most_similar("email"))
print(model.most_similar("morning"))






		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		





































