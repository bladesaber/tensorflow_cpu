from __future__ import print_function

from functools import reduce
from keras.utils.data_utils import get_file
import keras
from keras.preprocessing.sequence import pad_sequences
import tarfile
import numpy
import re

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

x, xq, y = vectorize_stories(train, word_idx, story_maxlen, query_maxlen)
tx, txq, ty = vectorize_stories(test, word_idx, story_maxlen, query_maxlen)

print('vocab = {}'.format(vocab))
print('x.shape = {}'.format(x.shape))
print('xq.shape = {}'.format(xq.shape))
print('y.shape = {}'.format(y.shape))
print('story_maxlen, query_maxlen = {}, {}'.format(story_maxlen, query_maxlen))

print('Build model...')

RNN = keras.layers.SimpleRNN
#RNN = keras.layers.LSTM
#RNN = keras.layers.GRU
EMBED_HIDDEN_SIZE = 50
SENT_HIDDEN_SIZE = 100
QUERY_HIDDEN_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 40
print('RNN / Embed / Sent / Query = {}, {}, {}, {}'.format(RNN,
                                                           EMBED_HIDDEN_SIZE,
                                                           SENT_HIDDEN_SIZE,
                                                           QUERY_HIDDEN_SIZE))

sentence = keras.layers.Input(shape=(story_maxlen,), dtype='int32')
# 这里将 embeding 层 并合到神经网中一同训练
encoded_sentence = keras.layers.Embedding(vocab_size, EMBED_HIDDEN_SIZE)(sentence)
encoded_sentence = keras.layers.Dropout(0.3)(encoded_sentence)

question = keras.layers.Input(shape=(query_maxlen,), dtype='int32')
encoded_question = keras.layers.Embedding(vocab_size, EMBED_HIDDEN_SIZE)(question)
encoded_question = keras.layers.Dropout(0.3)(encoded_question)

encoded_question = RNN(EMBED_HIDDEN_SIZE)(encoded_question)
encoded_question = keras.layers.RepeatVector(story_maxlen)(encoded_question)

merged = keras.layers.add([encoded_sentence, encoded_question])
merged = RNN(EMBED_HIDDEN_SIZE)(merged)
merged = keras.layers.Dropout(0.3)(merged)
preds = keras.layers.Dense(vocab_size, activation='softmax')(merged)

model = keras.Model(input = [sentence, question], outputs=preds)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

print('Training')
model.fit([x, xq], y,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=2,
          validation_split=0.05)
loss, acc = model.evaluate([tx, txq], ty,batch_size=BATCH_SIZE)
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))

