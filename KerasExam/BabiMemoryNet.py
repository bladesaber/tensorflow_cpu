from __future__ import print_function

from functools import reduce
from keras.utils.data_utils import get_file
import keras
from keras.preprocessing.sequence import pad_sequences
from keras import Sequential
from keras import layers
import tarfile
import numpy
import re

'''Trains a memory network on the bAbI dataset.
用于克服 Rnn 或 Lstm 的信息向量的限制
'''

try:
    path = get_file('D:/DataSet/babi-tasks-v1-2.tar.gz',
                    origin='https://s3.amazonaws.com/text-datasets/''babi_tasks_1-20_v1-2.tar.gz')
except:
    print('Error downloading dataset, please download it manually:\n'
          '$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2'
          '.tar.gz\n'
          '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
    raise

def tokenize(sentence):
    '''Return the tokens of a sentence including punctuation.
       >>> tokenize('Bob dropped the apple. Where is the apple?')
       ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
       '''
    return [x.strip() for x in re.split('(\W+)?', sentence) if x.strip()]

def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format
    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()

        # 分割1次
        nid, line = line.split(' ', 1)

        nid = int(nid)
        #print('nid: ',nid,' line: ',line)
        if nid == 1:
            story = []
        if '\t' in line:
            question, answer, supporting = line.split('\t')
            question = tokenize(question)
            if only_supporting:
                # Only select the related substory
                # 对 supporting.split() 每个元素做 int 处理
                supporting = map(int, supporting.split())
                # supporting 指答案句的位置
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, question, answer))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data

def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file, retrieve the stories and then convert the sentences into a single story.
    If max_length is supplied any stories longer than max_length tokens will be discarded.
    '''
    # 将 substory 平坦化
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data
            if not max_length or len(flatten(story)) < max_length]
    return data

def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    xs = []
    xqs = []
    ys = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        # let's not forget that index 0 is reserved
        # one hot 化
        y = numpy.zeros(len(word_idx) + 1)
        y[word_idx[answer]] = 1
        xs.append(x)
        xqs.append(xq)
        ys.append(y)
    # pad_sequences 相当于将不等长的 input 等长处理
    return (pad_sequences(xs, maxlen=story_maxlen),pad_sequences(xqs, maxlen=query_maxlen), numpy.array(ys))

# Default QA1 with 1000 samples
challenge = 'tasks_1-20_v1-2/en/qa1_single-supporting-fact_{}.txt'
# QA1 with 10,000 samples
# challenge = 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt'
# QA2 with 1000 samples
#challenge = 'tasks_1-20_v1-2/en/qa2_two-supporting-facts_{}.txt'
# QA2 with 10,000 samples
# challenge = 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt'
with tarfile.open(path) as tar:
    train = get_stories(tar.extractfile(challenge.format('train')))
    test = get_stories(tar.extractfile(challenge.format('test')))

# 获取词表
vocab = set()
for story, q, answer in train + test:
    vocab |= set(story + q + [answer])
vocab = sorted(vocab)

vocab_size = len(vocab) + 1
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
# 获取当前 story 的最大长度
story_maxlen = max(map(len, (x for x, _, _ in train + test)))
# 获取当前 question 的最大长度
query_maxlen = max(map(len, (x for _, x, _ in train + test)))

inputs_train, queries_train, answers_train = vectorize_stories(train, word_idx, story_maxlen, query_maxlen)
inputs_test, queries_test, answers_test = vectorize_stories(test, word_idx, story_maxlen, query_maxlen)

print('-')
print('Vocab size:', vocab_size, 'unique words')
print('Story max length:', story_maxlen, 'words')
print('Query max length:', query_maxlen, 'words')
print('Number of training stories:', len(train))
print('Number of test stories:', len(test))
print('-')
print('Here\'s what a "story" tuple looks like (input, query, answer):')
print(train[0])
print('-')
print('Vectorizing the word sequences...')

print('-')
print('inputs: integer tensor of shape (samples, max_length)')
print('inputs_train shape:', inputs_train.shape)
print('inputs_test shape:', inputs_test.shape)
print('-')
print('queries: integer tensor of shape (samples, max_length)')
print('queries_train shape:', queries_train.shape)
print('queries_test shape:', queries_test.shape)
print('-')
print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
print('answers_train shape:', answers_train.shape)
print('answers_test shape:', answers_test.shape)
print('-')
print('Compiling...')

print('Build model...')

# placeholders
input_sequence = keras.layers.Input((story_maxlen,))
question = keras.layers.Input((query_maxlen,))

# encoders
# embed the input sequence into a sequence of vectors
input_encoder_m = keras.Sequential()
input_encoder_m.add(keras.layers.Embedding(input_dim=vocab_size,output_dim=64))
input_encoder_m.add(keras.layers.Dropout(0.3))
# output: (samples, story_maxlen, embedding_dim)

# embed the input into a sequence of vectors of size query_maxlen
input_encoder_c = Sequential()
input_encoder_c.add(layers.Embedding(input_dim=vocab_size,output_dim=query_maxlen))
input_encoder_c.add(layers.Dropout(0.3))
# output: (samples, story_maxlen, query_maxlen)

# embed the question into a sequence of vectors
question_encoder = Sequential()
question_encoder.add(layers.Embedding(input_dim=vocab_size,output_dim=64,input_length=query_maxlen))
question_encoder.add(layers.Dropout(0.3))
# output: (samples, query_maxlen, embedding_dim)

# encode input sequence and questions (which are indices)
# to sequences of dense vectors
input_encoded_m = input_encoder_m(input_sequence)
input_encoded_c = input_encoder_c(input_sequence)
question_encoded = question_encoder(question)

# compute a 'match' between the first input vector sequence and the question vector sequence
# shape: `(samples, story_maxlen, query_maxlen)`
match = layers.dot([input_encoded_m, question_encoded], axes=(2, 2))
match = layers.Activation('softmax')(match)

# add the match matrix with the second input vector sequence
response = layers.add([match, input_encoded_c])  # (samples, story_maxlen, query_maxlen)
# 维度重排
response = layers.Permute((2, 1))(response)  # (samples, query_maxlen, story_maxlen)

# concatenate the match matrix with the question vector sequence
answer = layers.concatenate([response, question_encoded])

# the original paper uses a matrix multiplication for this reduction step.
# we choose to use a RNN instead.
answer = layers.LSTM(32)(answer)  # (samples, 32)

# one regularization layer -- more would probably be needed.
answer = layers.Dropout(0.3)(answer)
answer = layers.Dense(vocab_size)(answer)  # (samples, vocab_size)
# we output a probability distribution over the vocabulary
answer = layers.Activation('softmax')(answer)

# build the final model
model = keras.Model([input_sequence, question], answer)
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
