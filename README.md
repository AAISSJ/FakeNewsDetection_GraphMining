# GraphMining

- DAI5019 Graph Mining and Learning (Fall, 2022)


## 개요

세계적으로 국내 신종 코로나바이러스 감염증(COVID-19·코로나19) 확산세가 잦아들지 않고 있는 가운데 코로나19에 대한 가짜 뉴스가 마치 사실인 것처럼 소셜 네트워크 서비스(SNS)를 통해 확산하는 사례가 계속해서 보고되고 있다. 최근에는 특히 코로나19 백신과 관련해 가짜 뉴스가 심각한 문제로 떠올랐다. 가짜뉴스, 허위조작정보를 통해 예방 접종 거부를 조장하여 집단 면역체계 구축을 방해하는 행위는 인류의 생명과 안전을 위협하는 행위다.

따라서 위의 문제를 해결하기 위해, Graph Neural Network 기법을 사용하여 COVID-19 관련 가짜뉴스의 진위 여부를 판별할 수 있는 Node Classification 모델을 구축하고자 한다. 


## Data Preprocessing 

Sentence transformers에서 제공하는 pretrained model를 이용하여 train과 test data에 대해 768차원의 embedding 값을 얻을 수 있었다. pretrained model로는 ‘paraphrase-distilroberta-base-v1’와 ‘all-mpnet-base-v2’를 사용하였다. 

다음으로, 제공된 데이터셋에 있는 keyword(keybert_keywords, ner_keywords열)를 활용하여 edge feature로 사용하고자 Adjacency Matrix를 구축하였다. keybert_keywords와 ner_keywords 열에서 공통된 keyword를 공유하고 있는 claim끼리 연결하고 keyword가 몇 개나 겹치는지를 edge weight으로 두었다. 


## Analysis 

그래프의 node feature로는 코로나 관련 가짜 뉴스에 대한 Sentence Embedding을 사용하였다. 생성한 768차원의 Embedding이 각 문장이 가지고 있는 의미적/문법적 정보를 잘 함축하고 있는지, 또 문장 간의 차이를 잘 담고 있는지를 확인하기 위해서 t-SNE (t-distribution Stochastic Neighbor Embedding) 알고리즘을 적용하여 시각화했다.  t-SNE 는 고차원의 벡터로 표현되는 데이터 간의 neighbor structure를 보존하는 2 차원의 embedding vector를 학습함으로써, 고차원의 데이터를 2 차원의 지도로 표현하는 시각화 기법이다[1]. 아래 사진과 같이, True(1)값을 가지는 Node Feature들과 False(0)값을 가지는 Node Feature들이 2차원 상에서도 잘 구분되어 위치하는 것을 확인할 수 있다.

<div align="center">
<img width="406" alt="image" src="https://user-images.githubusercontent.com/76966915/211153712-5e7177ac-98fc-4399-aab9-ade2708bf751.png">
</div>

또한, Sentence Embedding 값만으로도 코로나 관련 가짜 뉴스의 진위 여부를 판별할 수 있는지 확인해보기 위해 Train data에 대해서 Machine Learning 알고리즘 등을 사용하여 Classification을 시도하였다. 사용한 Machine Learning 알고리즘은 MLP, Logistic Regression, SVM-RBF다. 실험 결과, Sentence Embedding 값만으로도 accuracy, recall, f1-score 모두 0.70가 넘는 것을 확인할 수 있었다. 즉, Node Feature로 사용한 Sentence Embedding이 그 자체만으로도 코로나 관련 가짜 뉴스의 진위 여부를 판별할 수 있는 유의미한 값임을 의미한다. 

<div align="center">
  <img width="208" alt="image" src="https://user-images.githubusercontent.com/76966915/211153737-acc75a23-a9d8-4288-b125-fff12edb2526.png">
  <img width="208" alt="image" src="https://user-images.githubusercontent.com/76966915/211153741-885a935f-6996-4cc9-82e4-4d2976c46ad2.png">
  <img width="208" alt="image" src="https://user-images.githubusercontent.com/76966915/211153742-4b947412-f70d-46ae-8940-4e41057574ff.png">
</div>

각 뉴스 간 연결되는 edge와 그 feature는 Keyword가 얼마나 겹치는지에 따라 설정하고자 했다. keybert_keywords, ner_keywords, keybert_keywords + ner_keywords 중 어떤 keyword를 반영하는지에 따라 Adjacency Matrix가 각기 다르게 나왔다. 먼저 서로 다른 데이터 간의 Adjacency Matrix를 생성하기 이전에 각 데이터의 keybert_keywords와 ner_keywords가 얼마나 유사한지 확인해보았다. 그 결과 train dataset과 teat dataset 전체에 대해서 keybert_keywords와 ner_keywords는 평균적으로 2.5372개로 같은 keyword들을 공유하고 있다. 따라서 대체로 keybert_keywords와 ner_keywords는 동일한 문서에 대해 대체로 유사한 키워드를 추출한다. 또한 train dataset과 test dataset 모두에 대해 keybert_keywords와 ner_keywords로 활용하여 만든 Adjacency Matrix를 시각화했다. 그 결과 아래의 그림처럼, 특정 키워드를 가지고 있는 노드들의 이웃과 그와 비슷한 노드들의 이웃은 유사하다는 것을 확인할 수 있었다. (ex. 첫번째 컬럼의 점들과 1000번째 컬럼의 점들이 거의 동일)

<div align ="center">
  <img width="198" alt="image" src="https://user-images.githubusercontent.com/76966915/211153766-9ab65912-ddaf-42c5-9762-e6675a6f4d8f.png">
</div>
