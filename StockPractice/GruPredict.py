import csv
import os
import numpy
import datetime
import keras
from keras import layers
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

class DataManager(object):
    def __init__(self,filepath):
        self.filePath = filepath
        self.files = os.listdir(self.filePath)
        if 'SHSE.000001.csv' in self.files:
            self.files.remove('SHSE.000001.csv')

    def creatData_Method_1(self,number):
        self.index = numpy.array(list(csv.reader(open(self.filePath+'SHSE.000001.csv','r'))))
        self.date = list(map(lambda x:datetime.datetime.strptime(x,'%Y/%m/%d %H:%M'),self.index[1:,0]))

        effect_number = 0

        sequense = []
        answer = []
        for f in self.files:
            with open(self.filePath+f,'r') as f1:
                data = numpy.array(list(csv.reader(f1)))
                newData = numpy.zeros((len(data)-1,4),dtype=numpy.float32)
                date = list(map(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'),data[1:,0]))

                if date == self.date:
                    newData[:,:3] = numpy.round(numpy.array(data[1:,2:5],dtype=numpy.float32))
                    newData[:,3] = numpy.round(numpy.array(self.index[1:,2],dtype=numpy.float32))
                else:
                    print(f+' date error')
                    continue
            effect_number += 1
            for i in range(number,len(newData),1):
                sequense.append(newData[i-number:i])
                answer_tem = numpy.zeros((21,),dtype=numpy.float32)
                answer_tem[int( newData[i,0]+ 10)] = 1.0
                answer.append(answer_tem)
            if effect_number == 10:
                break
        print('effect stock number: ',effect_number)
        return numpy.array(sequense),numpy.array(answer)

    def creatData_Method_2(self,number):
        self.index = numpy.array(list(csv.reader(open(self.filePath+'SHSE.000001.csv','r'))))
        self.date = list(map(lambda x:datetime.datetime.strptime(x,'%Y/%m/%d %H:%M'),self.index[1:,0]))

        effect_number = 0

        sequense = []
        answer = []
        for f in self.files:
            with open(self.filePath+f,'r') as f1:
                data = numpy.array(list(csv.reader(f1)))
                date = list(map(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'),data[1:,0]))

                if date == self.date:
                    data_tem = numpy.array(data[1:,2:5],dtype=numpy.float32)
                    index_tem = numpy.array(self.index[1:,2],dtype=numpy.float32)

                    for i in range(number,len(data_tem)):
                        data_seg = numpy.zeros((number+1,4),dtype=numpy.float32)
                        data_seg[0] = 100
                        for j in range(number):
                            data_seg[j+1,:3] = data_seg[j,:3] * (100.0 + data_tem[i-number+j])/100.0
                            data_seg[j+1,3] = data_seg[j,3] * (100.0 + index_tem[i-number+j])/100.0
                        result = data_seg[-1][0] * (100.0 + data_tem[i][0])/100.0
                        sequense.append(data_seg)
                        answer.append(result)
                else:
                    print(f+' date error')
                    continue

            effect_number += 1
            #if effect_number == 10:
            #    break
        return numpy.array(sequense),numpy.array(answer)

class NetWork(object):
    def train_cate(self,number):
        model = keras.Sequential()
        model.add(
            layers.GRU(units=100,use_bias=False,input_shape=(number,4),return_sequences=True,name='GRU_1')
        )
        model.add(layers.Dropout(0.3))
        model.add(
            layers.GRU(100,use_bias=True,return_sequences=False,name='GRU_2')
        )
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(units=21,use_bias=True,name='DENSE_1'))
        model.add(layers.Activation('softmax'))

        model.summary()

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        dataManager = DataManager('D:/DataSet/stock/preparation/')
        sequense, answer = dataManager.creatData_Method_1(number)

        #index = numpy.arange(len(sequense))
        #numpy.random.shuffle(index)

        #sequense = sequense[index]
        #answer = answer[index]

        #train_x = sequense[:int(len(sequense)*0.8)]
        #train_y = answer[:int(len(sequense) * 0.8)]
        #test_x = sequense[int(len(sequense) * 0.8):]
        #test_y = answer[int(len(sequense) * 0.8):]

        train_x = sequense
        train_y = answer

        #print('train epoch: ',len(train_x),' test epoch: ',len(test_x))

        Log_dir = 'D:/Git/tensorflow_cpu/StockPractice/'
        file_name = 'stock_GRU_GRU_DENSE'
        tb = TensorBoard(log_dir=Log_dir + file_name, write_grads=True)
        
        model.fit(train_x, train_y,
                  batch_size=256,
                  shuffle=True,
                  epochs=10,
                  verbose=1)
                  #callbacks=[tb],
                  #validation_data=[test_x,test_y])

        #model.save(filepath=Log_dir+'stock_GRU_GRU_DENSE.h5',overwrite=True)

    def train_point(self, number):

        model = keras.Sequential()
        model.add(
            layers.GRU(units=100, use_bias=False, input_shape=(number+1, 4), return_sequences=True, name='GRU_1')
        )
        model.add(layers.Dropout(0.3))
        model.add(
            layers.GRU(100, use_bias=True, return_sequences=False, name='GRU_2')
        )
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(units=1, use_bias=True, name='DENSE_1'))

        model.summary()

        model.compile(optimizer='adam',
                      loss='mse',
                      metrics=['accuracy'])

        dataManager = DataManager('D:/DataSet/stock/preparation/')
        train_x, train_y = dataManager.creatData_Method_2(number)

        index = numpy.arange(len(train_x))
        numpy.random.shuffle(index)

        train_x = train_x[index]
        train_y = train_y[index]
        print('train epoch: ', len(train_x))

        Log_dir = 'D:/Git/tensorflow_cpu/StockPractice/'
        file_name = 'stock_GRU_GRU_DENSE'
        tb = TensorBoard(log_dir=Log_dir + file_name, write_grads=True)

        model.fit(train_x, train_y,
                  batch_size=256,
                  shuffle=True,
                  epochs=5,
                  verbose=1)
        # callbacks=[tb],
                #validation_data=[test_x,test_y])

        model.save(filepath=Log_dir+'stock_GRU_GRU_DENSE_point.h5',overwrite=True)

        testManager = DataManager('D:/DataSet/stock/test/')
        test_x,test_y = testManager.creatData_Method_2(number)

        print(' test epoch: ', len(test_x))

        predict_y = model.predict_on_batch(test_x)
        plt.plot(range(len(test_y)),test_y,'-')
        plt.plot(range(len(test_y)), predict_y, '-')
        plt.show()

#netWork = NetWork()
#netWork.train_point(100)

testManager = DataManager('D:/DataSet/stock/test/')
test_x,test_y = testManager.creatData_Method_2(100)
model = keras.models.load_model('D:/Git/tensorflow_cpu/StockPractice/stock_GRU_GRU_DENSE_point.h5')
print(' test epoch: ', len(test_x))

predict_y = model.predict_on_batch(test_x)
plt.plot(range(len(test_y)),test_y,'-',label='test')
plt.plot(range(len(test_y)), predict_y, '-',label='predict')
plt.show()
