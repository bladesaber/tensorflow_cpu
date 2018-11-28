from keras.models import Sequential
from keras import layers
import numpy

'''
this model is used to make the machine learn how to solute the add function
'''

class CharacterTable(object):

    def __init__(self,chars):
        self.chars = sorted(set(chars))

        # char_indices char转义到indice 的字典
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        # char_indices indice转义到char 的字典
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        x = numpy.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in x)

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

TRAINING_SIZE = 50000
DIGITS = 3
REVERSE = False

# Maximum length of input is 'int + int' (e.g., '345+678'). Maximum length of
# int is DIGITS.
MAXLEN = DIGITS + 1 + DIGITS

chars = '0123456789+ '
ctable = CharacterTable(chars)

questions = []
expected = []
seen = set()
print('Generating data...')

while len(questions) < TRAINING_SIZE:
    f = lambda: int(''.join(numpy.random.choice(list('0123456789')) for i in range(numpy.random.randint(1, DIGITS + 1))))
    a, b = f(), f()
    key = tuple(sorted((a, b)))
    if key in seen:
        continue
    seen.add(key)
    q = '{}+{}'.format(a, b)

    # 长度规整
    query = q + ' ' * (MAXLEN - len(q))

    ans = str(a + b)

    # 长度规整
    ans += ' ' * (DIGITS + 1 - len(ans))
    if REVERSE:
        # Reverse the query, e.g., '12+345  ' becomes '  543+21'. (Note the space used for padding.)
        query = query[::-1]
    questions.append(query)
    expected.append(ans)
print('Total addition questions:', len(questions))

'''
for i in range(len(questions)):
    print('-',questions[i],'-')
    print('-',expected[i],'-')
    print('-----')
'''

print('Vectorization...')
x = numpy.zeros((len(questions), MAXLEN, len(chars)), dtype=numpy.bool)
y = numpy.zeros((len(questions), DIGITS + 1, len(chars)), dtype=numpy.bool)
for i, sentence in enumerate(questions):
    x[i] = ctable.encode(sentence, MAXLEN)
    # print(sentence)
    # print(x[i])
for i, sentence in enumerate(expected):
    y[i] = ctable.encode(sentence, DIGITS + 1)

indices = numpy.arange(len(y))
numpy.random.shuffle(indices)
x = x[indices]
y = y[indices]

split_at = len(x) - len(x) // 10
(x_train, x_val) = x[:split_at], x[split_at:]
(y_train, y_val) = y[:split_at], y[split_at:]

print('Training Data:')
print(x_train.shape)
print(y_train.shape)

print('Validation Data:')
print(x_val.shape)
print(y_val.shape)

hidden_size = 128
batch_size = 128
layer_structer = 1

#RNN = layers.LSTM
RNN = layers.SimpleRNN

print('Build model...')
model = Sequential()

# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE.
# Note: In a situation where your input sequences have a variable length, use input_shape=(None, num_feature).
model.add(RNN(hidden_size, input_shape=(MAXLEN, len(chars))))

# As the decoder RNN's input, repeatedly provide with the last output of RNN for each time step.
# Repeat 'DIGITS + 1' times as that's the maximum length of output, e.g., when DIGITS=3, max output is 999+999=1998.
# 将第一维变量重复n次
model.add(layers.RepeatVector(DIGITS + 1))

for _ in range(layer_structer):
    # By setting return_sequences to True, return not only the last output but
    # all the outputs so far in the form of (num_samples, timesteps,
    # output_dim). This is necessary as TimeDistributed in the below expects the first dimension to be the timesteps.
    # return_sequences 获取每个序列的输出
    model.add(RNN(hidden_size, return_sequences=True))

model.add(layers.Dense(len(chars), activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

for iteration in range(1, 20):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=1,
              validation_data=(x_val, y_val))

    for i in range(10):
        ind = numpy.random.randint(0, len(x_val))
        rowx, rowy = x_val[numpy.array([ind])], y_val[numpy.array([ind])]
        preds = model.predict_classes(rowx, verbose=0)
        q = ctable.decode(rowx[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print('Q', q[::-1] if REVERSE else q, end=' ')
        print('T', correct, end=' ')
        if correct == guess:
            print(colors.ok + '☑' + colors.close, end=' ')
        else:
            print(colors.fail + '☒' + colors.close, end=' ')
        print(guess)
