## Pytorch-POS-tagging
POS tagging model with LSTM and Pytorch.

## Data
NLTK data

## Purpose
LSTM tagger를 만들고 NLTK 데이터를 학습시켜 효과적으로 품사를 구분하는 모델을 구현한다.  

## Thoughts
Pytorch LSTM 모델에 맞게 데이터를 준비하고 (vocab 생성, tag_to_ix, word_to_ix 등 사전 만들기)  
적절한 Hyperparameters를 설정한 뒤(embedding_dim, hidden_dim 등) 모델을 학습시킨다.  
Pytorch LSTM 구조를 잘 이해하는 것이 주요 목적이며, 이를 위해 코드에 자세하게 주석을 달아놓았다.  
추후에 참고가 가능할 것이다.

## Result
테스트 문장: 'the director is 61 years old'  
실제 답: the - DT, director - NN, is - VBZ, 61 - CD, years - NMS, old - JJ  
아웃풋: DT, NN, VBZ, CD, NNS, JJ  


