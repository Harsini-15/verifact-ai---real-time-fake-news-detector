import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
import pickle
import os

class LSTMModel:
    def __init__(self, max_words=5000, max_len=100):
        self.max_words = max_words
        self.max_len = max_len
        self.model = self._build_model()
        self.tokenizer = Tokenizer(num_words=self.max_words, lower=True)

    def _build_model(self):
        model = Sequential()
        model.add(Embedding(self.max_words, 128))
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self, texts, labels, epochs=5, batch_size=32):
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=self.max_len)
        y = np.array(labels)
        
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
        
        # Save tokenizer and model
        os.makedirs('models', exist_ok=True)
        with open('models/lstm_tokenizer.pkl', 'wb') as f:
            pickle.dump(self.tokenizer, f)
        self.model.save('models/lstm_model.h5')

    def predict(self, text):
        # Load model and tokenizer if not already loaded
        if not hasattr(self, 'loaded_model'):
            self.loaded_model = tf.keras.models.load_model('models/lstm_model.h5')
            with open('models/lstm_tokenizer.pkl', 'rb') as f:
                self.loaded_tokenizer = pickle.load(f)
        
        seq = self.loaded_tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=self.max_len)
        prediction = self.loaded_model.predict(padded)
        return float(prediction[0][0])
