#################
# Dataset Intro #
#################

# 데이터셋 : Naver Sentiment Movie Corpus(https://github.com/e9t/nsmc/)
'''
>> 네이버 영화 리뷰 중 영화당 100개의 리뷰를 모아 총 200,000개의 리뷰(훈련 : 15만개, 테스트 :5만개)로 이루어져있고,
>> 네이버 영화 리뷰 중 영화당 100개의 리뷰를 모아 총 200,000개의 리뷰(훈련 : 15만개, 테스트 :5만개)로 이루어져있고,

>> 데이터는 id, document, label 세개의 열로 이루어져 있음
>> id : 리뷰의 고유한 Key값
>> document : 리뷰의 내용
>> label : 긍정(1)인지 부정(0)인지 나타냄, 평점이 긍정(9~10점), 부정(1~4점), 5~8점은 제거
'''

import json
import os
import nltk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, rc
from pprint import pprint
from konlpy.tag import Okt
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics


#############
# File Open #
#############

# *.txt 파일에서 데이터를 불러오는 메서드
def read_data(filename):
    with open(filename, 'r', encoding='UTF-8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]  # 제목열 제외
    return data


# nsmc 데이터를 불러와서 python 변수에 담기
train_data = read_data('ratings_train.txt')  # 트레이닝 데이터 Open
test_data = read_data('ratings_test.txt')  # 테스트 데이터 Open

print(len(train_data))
print(len(test_data))
print(train_data[0])
print(test_data[0])

#################
# PreProcessing #
#################

'''
데이터를 학습하기에 알맞게 처리해보자. konlpy 라이브러리를 사용해서 형태소 분석 및 품사 태깅을 진행한다.
네이버 영화 데이터는 맞춤법이나 띄어쓰기가 제대로 되어있지 않은 경우가 있기 때문에
정확한 분류를 위해서 konply를 사용한다.
konlpy에는 여러 클래스가 존재하지만 그 중 okt(open korean text)를 사용하여 간단한 문장분석을 실행한다.
'''
okt = Okt()


# print(okt.pos('이 밤 그날의 반딧불을 당신의 창 가까이 보낼께요'))

# Train, Test 데이터셋에 형태소 분석을 통해 품사태깅 작업 진행
def tokenize(doc):
    # norm은 정규화, stem은 근어로 표시하기를 나타냄
    return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)]


if os.path.isfile('train_docs.json'):
    with open('train_docs.json', 'r', encoding='UTF-8') as f:
        train_docs = json.load(f)
    with open('test_docs.json', 'r', encoding='UTF-8') as f:
        test_docs = json.load(f)
else:
    train_docs = [(tokenize(row[1]), row[2]) for row in train_data]
    test_docs = [(tokenize(row[1]), row[2]) for row in test_data]
    # Json 파일로 저장
    with open('train_docs.json', 'w', encoding='UTF-8') as make_file:
        json.dump(train_docs, make_file, ensure_ascii=False, indent='\t')
    with open('test_docs.json', 'w', encoding='UTF-8') as make_file:
        json.dump(test_docs, make_file, ensure_ascii=False, indent='\t')

# 전처리 작업 데이터 확인
pprint(train_docs[0])
pprint(test_docs[0])
