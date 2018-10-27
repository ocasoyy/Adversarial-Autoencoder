"""
Author: Youyoung
Reference: Pytorch Documents about NLP
"""


# Setting
import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


# 파이토치 LSTM의 인풋은 3D 텐서임
# (sequence_length, batch_size, input_size)

torch.manual_seed(1)


# (1, 3) size를 가진 5개의 텐서 생성
# 즉, 길이 5의 sequence 생성
inputs = [torch.randn(1, 3) for _ in range(5)]

# hidden state 와 cell state 초기화
hidden, cell = torch.randn(1, 1, 3), torch.randn(1, 1, 3)


# input들을 stack해서 한 번에 집어 넣자
inputs = torch.cat(inputs, dim=0).view(len(inputs), 1, -1)    # (5, 1, 3)

# LSTM 빌드: Inputs: inputs, (h_0, c_0)
# Parameters
# input_size=3, hidden_size=3 (number of features in the input x and hidden state h)
# num_layers=1 (number of recurrent layers, 1 이상이면 stacked LSTM이란 소리)
# bias=True
# bidirectional=False
lstm = nn.LSTM(input_size=3, hidden_size=3) # input_dim = 3, output_dim=3

# Instance method
# inputs' shape: (sequence_length, batch_size, input_size)
# h_o: tensor containing the initial hidden state, shape = (num_layers*num_directions, batch_size, hidden_size)
# c_o: tensor containing the initial cell state, shape = (num_layers*num_directions, batch_size, hidden_size)
out, hidden = lstm(inputs, (hidden, cell))

# Example1: LSTM for Part-of-Speech Tagging
# link: http://pytorch.kr/beginner/nlp/sequence_models_tutorial.html#sphx-glr-beginner-nlp-sequence-models-tutorial-py
# take the log softmax of the affine map of the hidden state,
# the predicted tag is the tag that has the max value in this vector

training_data = [("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
                 ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])]

tag_to_ix = {"DET": 0, "NN": 1, "V": 2}
word_to_ix = {}

for sentence, _ in training_data:
    for word in sentence:
        # 사전의 key에 word가 없으면 추가
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

print(word_to_ix)


# to_ix를 이용하여 단어들을 vocab 위치에 맞게 integer로 변형시키기
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)



# POS 태깅에서 보통 embedding_dim과 hidden_dim은 32나 64를 많이 쓰지만
# 여기서는 간단한 시범을 위해 6으로 한정하겠다.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6
BATCH_SIZE = 1


# 모델 빌드
class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, batch_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        # LSTM takes word embeddings as inputs, and outputs hidden states with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden_to_tag = nn.Linear(hidden_dim, tagset_size)

        # hidden state와 cell state 초기화
        self.hidden = self.init_hidden()


    def init_hidden(self):
        return (torch.zeros(1, self.batch_size, self.hidden_dim), torch.zeros(1, self.batch_size, self.hidden_dim))


    def forward(self, sentence):
        # 여기서 sentence는 embeds(Embedding Matrix)를 만드는데 쓰이는 Input이며
        # 이 Input은 word_to_ix를 이용하여 index를 표시한 단어 벡터 리스트이다. (Variable(tensor))
        embeds = self.embeddings(sentence)    # shape = (examples=len(sentence), input)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), self.batch_size, -1), self.hidden)

        # 마지막에 flatten함
        tag_space = self.hidden_to_tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores



# 학습
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix), BATCH_SIZE)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


def train_model(epochs=10):
    for epoch in range(epochs):
        total_loss = 0

        for sentence, tags in training_data:
            model.zero_grad()

            # Step 1. hidden state of the LSTM 초기화 -- detaching it from its history on the last instance.
            model.hidden = model.init_hidden()

            # Step 2. Get our inputs ready for the network, that is, turn them into Tensors of word indices.
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix)

            # Step 3. Run our forward pass.
            tag_scores = model(sentence_in)

            # Step 4. Compute the loss, gradients, and update the parameters
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            loss_avg = float(total_loss / len(data))
            print("Epoch: {} loss {:.2f}".format(epoch, loss_avg))




# NLTK 데이터로 해보자
tagged_sentences = nltk.corpus.treebank.tagged_sents()

print(tagged_sentences[0])
print("Tagged sentences: ", len(tagged_sentences))                # 3914
print("Tagged words:", len(nltk.corpus.treebank.tagged_words()))  # 100676

# 위 training_data와 같은 형태로 데이터셋 구성
total = []
for i in range(len(tagged_sentences)):
    word_list = []
    tag_list = []

    for index in range(len(tagged_sentences[i])):
        word, tag = tagged_sentences[i][index]
        word_list.append(word)
        tag_list.append(tag)

    total.append((word_list, tag_list))


assert len(tagged_sentences) == len(total), "Dataset wrong"



# to_ix를 이용하여 단어들을 vocab 위치에 맞게 integer로 변형시키기
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)



# 데이터셋 세부 준비
# sentence_in을 위한 준비
all_sentences = []

for i in range(len(total)):
    sentence = ' '.join(total[i][0])
    all_sentences.append(sentence)

text = ' '.join(all_sentences).split()
vocab = set(text)
vocab_size = len(vocab)
word_to_ix = {word: index for index, word in enumerate(vocab)}


# targets를 위한 준비
all_tags = []

for i in range(len(total)):
    tag = ' '.join(total[i][1])
    all_tags.append(tag)

tag_set = set(' '.join(all_tags).split())
tag_to_ix = {tag: index for index, tag in enumerate(tag_set)}


# 모델 빌드
class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, batch_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        # LSTM takes word embeddings as inputs, and outputs hidden states with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden_to_tag = nn.Linear(hidden_dim, tagset_size)

        # hidden state와 cell state 초기화
        self.hidden = self.init_hidden()


    def init_hidden(self):
        return (torch.zeros(1, self.batch_size, self.hidden_dim), torch.zeros(1, self.batch_size, self.hidden_dim))


    def forward(self, sentence):
        # 여기서 sentence는 embeds(Embedding Matrix)를 만드는데 쓰이는 Input이며
        # 이 Input은 word_to_ix를 이용하여 index를 표시한 단어 벡터 리스트이다. (Variable(tensor))
        embeds = self.embeddings(sentence)    # shape = (examples=len(sentence), input)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), self.batch_size, -1), self.hidden)

        # 마지막에 flatten함
        tag_space = self.hidden_to_tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores




# 학습 진행
# embedding_dim, hidden_dim, vocab_size, tagset_size, batch_size
model = LSTMTagger(32, 32, len(word_to_ix), len(tag_to_ix), 1)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


def train_model(epochs=10):
    for epoch in range(epochs):
        total_loss = 0

        for sentence, tags in total:
            model.zero_grad()

            # Step 1. hidden state of the LSTM 초기화 -- detaching it from its history on the last instance.
            model.hidden = model.init_hidden()

            # Step 2. Get our inputs ready for the network, that is, turn them into Tensors of word indices.
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix)

            # Step 3. Run our forward pass.
            tag_scores = model(sentence_in)

            # Step 4. Compute the loss, gradients, and update the parameters
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()

        # if epoch % 5 == 0:
        loss_avg = float(total_loss / len(total))
        print("Epoch: {} loss {:.2f}".format(epoch, loss_avg))

# Run
train_model(epochs=2)


# Test
from nltk import word_tokenize
sent = "the director is 61 years old"

# 실제 답
nltk.pos_tag(word_tokenize(sent))


test_sentence = "the director is 61 years old".split()
inp = prepare_sequence(test_sentence, word_to_ix)
result = model(inp)

# 모델이 반환한 답
for i in range(len(test_sentence)):
    idx = torch.argmax(result[i])
    print(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(idx)])














# ----------------------------------------------------------------------
# 개념을 알기 위한 Simple Example(Pytorch Document에 있음)

# Setting
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


# Word Embeddings
# 임베딩 시 각 단어가 참조할 Index를 정의해야 함: lookup table의 key
# (V, D) 행렬: V = Vocab size, D = 임베딩 벡터 차원
# torch.nn.Embedding(num_embeddings=V, embedding_dim=D), D는 하이퍼파라미터

torch.manual_seed(1)

word_to_ix = {'hello':0, 'world':1}
embeds = nn.Embedding(num_embeddings=2, embedding_dim=5, sparse=False)

# torch.Tensor는 tensor와 empty의 mixture 같음.
# data를 피드할 때 Tensor는 global default dtype을 쓰고 tensor는 data의 data type을 참조함
lookup_tensor = torch.tensor([word_to_ix['hello']], dtype=torch.long)

# lookup table에서 lookup tensor를 찾아 꺼내서 임베딩하기
hello_embed = embeds(lookup_tensor)
print(hello_embed)


# Example1: N-Gram Language Modeling
CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

# We will use Shakespeare Sonnet 2: length = 115

test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()


# build a list of tuples.
# Each tuple is ([ word_i-2, word_i-1 ], target word)
trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
            for i in range(len(test_sentence) - 2)]

# 사전 만들기
vocab = set(test_sentence)
vocab_size = len(vocab)
word_to_ix = {word: index for index, word in enumerate(vocab)}


# 모델 빌드
class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        # 먼저 임베딩을 해준다. (97, 10)
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=EMBEDDING_DIM)
        self.linear1 = nn.Linear(in_features=context_size*embedding_dim, out_features=128) # in: 20
        self.linear2 = nn.Linear(in_features=128, out_features=vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))    # 쫙 다 펴준다.
        out = F.relu(self.linear1(embeds))                # Relu 한 번 거치고
        out = self.linear2(out)
        log_probs = F.log_softmax(input=out, dim=1)       # log_softmax 이용
        return log_probs


model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
losses = []
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(10):
    total_loss = torch.Tensor([0])
    for context, target in trigrams:

        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # context에 들어 있는 2개의 단어를 integer index로 바꾸고
        # 텐서로 만들어준다.
        # 예: context_idxs = tensor([64, 71]): word_to_ix에서 64번째, 71번째에 잇는 단어
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old instance
        model.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next words
        # 이제 이걸 임베딩 시키면
        # embeds = embeddings(context_idxs).view((1, -1))이 실행되는데
        # 예: embeds = tensor([[20개의 숫자]]), size: (1, 20)
        # 왜냐하면 2개의 단어가 각각 10차원의 임베딩 벡터로 변환되었기 때문
        # 이러한 embeds가 len(triagrams)만큼 있는 셈이니까 전체 shape = (113, 20)
        log_probs = model(context_idxs)

        # Step 4. Compute your loss function.
        # (Again, Torch wants the target word wrapped in a variable)
        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        # Get the Python number from a 1-element Tensor by calling tensor.item()
        total_loss += loss.item()
    losses.append(total_loss)
print(losses)  # The loss decreased every iteration over the training data!




# Example2: Continuous Bag-of-Words
# sentence가 있으면 그 sentence 내에 있는 token의 순서는 무시하고
# 단순히 그 토큰 벡터들을 평균을 낸다. (단지 1개의 node in DAG)
# 굉장히 효율적이다. FastText를 쓰자. (gensim)
# CBOW는 sequential하지 않으며 확률 문제와도 상관 없다.

CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right

raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word: index for index, word in enumerate(vocab)}

# data 채우기
data = []

for i in range(CONTEXT_SIZE, len(raw_text) - CONTEXT_SIZE):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))
print(data[:5])


# 모델 빌드
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.linear1 = nn.Linear(in_features=2*context_size*embedding_dim, out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=vocab_size)


    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))    # (1, 40)
        out = self.linear1(embeds)
        out = F.relu(out)
        out = self.linear2(out)
        log_probs = F.log_softmax(input=out, dim=1)
        return log_probs


    def make_context_vector(self, context, word_to_ix):
        word_ix = [word_to_ix[word] for word in context]
        return torch.tensor(word_ix, dtype=torch.long)


    def make_target_vector(self, target, word_to_ix):
        word_ix = [word_to_ix[target]]
        return torch.tensor(word_ix, dtype=torch.long)



# model, loss, optimizer
cbow = CBOW(vocab_size=vocab_size, embedding_dim=10, context_size=CONTEXT_SIZE)
loss_func = nn.NLLLoss()
optimizer = torch.optim.Adam(cbow.parameters(), lr=0.001)

epochs=30

for epoch in range(epochs):
    total_loss = 0

    for context, target in data:
        #input_variable = Variable(torch.tensor([word_to_ix[word] for word in context]))
        #target_variable = Variable(torch.tensor([word_to_ix[target]]))
        input_variable = Variable(cbow.make_context_vector(context=context, word_to_ix=word_to_ix))
        target_variable = Variable(cbow.make_target_vector(target=target, word_to_ix=word_to_ix))

        cbow.zero_grad()
        log_probs = cbow(input_variable)
        loss = loss_func(log_probs, target_variable)
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()

    if epoch % 5 == 0:
        loss_avg = float(total_loss / len(data))
        print("Epoch: {} loss {:.2f}".format(epoch, loss_avg))
