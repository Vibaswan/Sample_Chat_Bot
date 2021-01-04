from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import tensorflow
import json
import pickle
import random
import numpy
import nltk
# nltk.download()
from pprint import pprint
from nltk import WordNetLemmatizer
stemmer = WordNetLemmatizer()

with open('intents.json') as file:
    data = json.load(file)


# pprint(data)

words = []
labels = []
# docs = []
docs_x = []
docs_y = []
ignore_words = ['?', '!']

for intent in data['intents']:
    for pattern in intent['pattern']:
        # take each word and tokenize it
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent['tag'])
        # docs.append((wrds, intent['tag']))

    if intent["tag"] not in labels:
        labels.append(intent['tag'])

print('words == ' + str(words))
print('labels == ' + str(labels))
print('docs are === ' + str(docs_x) + ' ' + str(docs_y))


# lammatizzing or basically stemming and refactoring the data nicely
words = [stemmer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
labels = sorted(list(set(labels)))

# saving this to be used in interaction.py script
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(labels, open('classes.pkl', 'wb'))


print('\n\n\n\n After lammatizzing')
print('words == ' + str(words))
print('labels == ' + str(labels))


# forming the training data
training = []
output = []
output_empty = [0] * len(labels)

for x, doc in enumerate(docs_x):

    bag = []
    wrds = [stemmer.lemmatize(word.lower()) for word in doc]

    # fill the bag if the word is present or else 0
    for word in words:
        if word in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = list(output_empty)
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)

print('\n\nAfter training')
print('training === ' + str(training))
print('output ===== ' + str(output))

# building the model
model = Sequential()
model.add(Dense(128, input_shape=(len(training[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(output[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# fitting and saving the model
hist = model.fit(training, output, epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("model created")
