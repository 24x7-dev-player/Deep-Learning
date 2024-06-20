docs = ['go india',
		'india india',
		'hip hip hurray',
		'jeetega bhai jeetega india jeetega',
		'bharat mata ki jai',
		'kohli kohli',
		'sachin sachin',
		'dhoni dhoni',
		'modi ji ki jai',
		'inquilab zindabad']

from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()

tokenizer.fit_on_texts(docs)

len(tokenizer.word_index)

sequences = tokenizer.texts_to_sequences(docs)
sequences

from keras.utils import pad_sequences
sequences = pad_sequences(sequences,padding='post')
sequences

model = Sequential()
model.add(Embedding(17,output_dim=2,input_length=5))

model.summary()

model.compile('adam','accuracy')

pred = model.predict(sequences)
print(pred)

from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras import Sequential
from keras.layers import Dense,SimpleRNN,Embedding,Flatten

(X_train,y_train),(X_test,y_test) = imdb.load_data()

X_train = pad_sequences(X_train,padding='post',maxlen=50)
X_test = pad_sequences(X_test,padding='post',maxlen=50)

X_train.shape

model = Sequential()
model.add(Embedding(10000, 2,50))
model.add(SimpleRNN(32,return_sequences=False))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train,epochs=5,validation_data=(X_test,y_test))


#sencond Approach
import numpy as np

docs = ['go india',
		'india india',
		'hip hip hurray',
		'jeetega bhai jeetega india jeetega',
		'bharat mata ki jai',
		'kohli kohli',
		'sachin sachin',
		'dhoni dhoni',
		'modi ji ki jai',
		'inquilab zindabad']

from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(oov_token='<nothing>')

tokenizer.fit_on_texts(docs)

tokenizer.word_index

tokenizer.word_counts

tokenizer.document_count

sequences = tokenizer.texts_to_sequences(docs)
sequences

from keras.utils import pad_sequences

sequences = pad_sequences(sequences,padding='post')

sequences

from keras.datasets import imdb
from keras import Sequential
from keras.layers import Dense,SimpleRNN,Embedding,Flatten

(X_train,y_train),(X_test,y_test) = imdb.load_data()

X_train[0]

len(X_train[2])

X_train = pad_sequences(X_train,padding='post',maxlen=50)
X_test = pad_sequences(X_test,padding='post',maxlen=50)

X_train[0]

model = Sequential()

model.add(SimpleRNN(32,input_shape=(50,1),return_sequences=False))
model.add(Dense(1,activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X_train,y_train,epochs=5,validation_data=(X_test,y_test))

