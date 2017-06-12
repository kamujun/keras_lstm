
# coding: utf-8

# In[2]:


from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

from keras.utils import plot_model

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)

batch_size = 32


# In[3]:


from datetime import datetime
print(datetime.now().strftime("%Y/%m/%d %H:%M:%S"))


# In[4]:


print('Loading data...')


(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

x_train = sequence.pad_sequences(x_train)
x_test = sequence.pad_sequences(x_test)

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')


# In[5]:


# try using different optimizers and different optimizer configs
def create_model(activation='sigmoid', drop_out=0.2, recurrent_dropout=0.2):
    model = Sequential()
    model.add(Embedding(max_features, 128))
    model.add(LSTM(128, dropout=drop_out, recurrent_dropout=recurrent_dropout))
    model.add(Dense(1, activation=activation))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    # # plot model
    # plot_model(model, to_file='model.png', show_shapes=True)

    return model


# In[6]:


print(datetime.now().strftime("%Y/%m/%d %H:%M:%S"))


# In[6]:


# # train
# from matplotlib import pyplot as plt
# 
# print('Train...')
# lstm = model.fit(x_train, y_train,
#                  batch_size=batch_size,
#                  epochs=15,
#                  validation_data=(x_test, y_test))
# 
# # evaluate
# score, acc = model.evaluate(x_test, y_test,
#                             batch_size=batch_size)
# print('Test score:', score)
# print('Test accuracy:', acc)
# 
# # plot acc and loss
# x = range(15)
# plt.plot(x, lstm.history['acc'], label="acc")
# plt.plot(x, lstm.history['loss'], label="loss")
# 
# plt.title("binary train accuracy")
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.show()


# In[8]:


# grid search
#activations = ["softplus", "softsign", "relu", "tanh", "sigmoid", "hard_sigmoid", "linear"]
activations = ["tanh", "sigmoid"]

drop_outs = [0.2, 0.5]
recurrent_dropouts = [0.2, 0.5]

param_grid = dict(activation=activations, drop_out=drop_outs, recurrent_dropout=recurrent_dropouts)
model = KerasClassifier(build_fn=create_model, nb_epoch=15, batch_size=batch_size, verbose=0)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy')
grid_result = grid.fit(x_train, y_train)

print(grid_result.best_score_)
print(grid_result.best_params_)

print(datetime.now().strftime("%Y/%m/%d %H:%M:%S"))


# In[ ]:




