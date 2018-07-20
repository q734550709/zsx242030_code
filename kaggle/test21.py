#使用词袋法(Bag-of-Words)对示例进行特征向量化
#将上述两个句子以字符串的数据类型分别存储在变量sent1与send2
from sklearn.feature_extraction.text import CountVectorizer
import nltk

send1="The cat is walking in the bedroom."
send2="A dog was running across the kitchen."
count_vec=CountVectorizer()
sentences=[send1,send2]
print(count_vec.fit_transform(sentences).toarray())
print(count_vec.get_feature_names())
print("====================================")
#nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
#对句子进行词汇分割和正规化,有些情况如aren't需要分割为are和n't;
#或者I'm要分割为I和'm
tokens_1=nltk.word_tokenize(send1)
print(tokens_1)

tokens_2=nltk.word_tokenize(send2)
print(tokens_2)
print("====================================")
#整理两句的词表,并且按照ASCII的排序输出
vocab_1=sorted(set(tokens_1))
print(vocab_1)

vocab_2=sorted(set(tokens_2))
print(vocab_2)
print("====================================")
#初始化stemmer寻找各个词汇最原始的词根
stemmer=nltk.stem.PorterStemmer()
stem1=[stemmer.stem(t) for t in tokens_1]
print(stem1)

stem2=[stemmer.stem(t) for t in tokens_2]
print(stem2)
print("=====================================")

#初始化词性标注器,对每个词汇进行标注
pos_tag1=nltk.tag.pos_tag(tokens_1)
print(pos_tag1)
pos_tag2=nltk.tag.pos_tag(tokens_2)
print(pos_tag2)






































