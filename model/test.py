from pymongo import MongoClient
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import tensorflow as tf
import numpy as np
from konlpy.tag import Okt
import nltk

selected_words = []
model = None
okt = Okt()
all_count = 0
pos_count = 0

client = MongoClient('localhost', 27017)
db = client['local']
collection = db.get_collection('movie')
reply_list = []


def mongo_select_all():
    for one in collection.find({}, {'_id': 0, 'movieNm': 1, 'content': 1, 'score': 1}):
        reply_list.append(one['content'])
    return reply_list

mongo_select_all()
all_count = len(reply_list)
print('>> 영화 전체 댓글 수: {}'.format(all_count))

for i in reply_list:
    selected_words.append(okt.morphs(i))

selected_words = [t for d in selected_words for t in d]
# print(selected_words)
text = nltk.Text(selected_words, name='NSMC')
print(">> 영화 전체 단어 수: {}".format(len(text)))
print(">> 영화 전체 단어 수(중복제거): {}".format(len(set(text.tokens))))
print(">> 영화 단어 빈도수 많은 순으로 1~10위까지: {}".format(text.vocab().most_common(10)))

font_fname = 'C:\Windows\Fonts\gulim.ttc'
font_name = font_manager.FontProperties(fname=font_fname).get_name()
rc('font', family=font_name)

x_axis = [f[0] for f in text.vocab().most_common(50)]
y_axis = [f[1] for f in text.vocab().most_common(50)]
plt.figure(figsize=(20, 10))
plt.xticks(rotation = 60)
plt.bar(x_axis,y_axis)
plt.show()

