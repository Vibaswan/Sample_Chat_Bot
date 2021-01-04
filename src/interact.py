from keras.models import load_model
import pickle
import json
import nltk
import numpy
import random
from nltk import WordNetLemmatizer
stemmer = WordNetLemmatizer()


def bag_of_words(sentence, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(sentence)
    s_words = [stemmer.lemmatize(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def chat(words, labels, intents, model):
    print("\n\n\n\nStart talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        p = bag_of_words(inp, words)
        results = model.predict(numpy.array([p]))[0]
        # print(results)
        results_index = numpy.argmax(results)
        # print(results_index)
        tag = labels[results_index]
        if results[results_index] > 0.7:
            for tg in intents["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

            print(random.choice(responses))
        else:
            print('Sorry i did not get that!, try again')


def main():
    words = pickle.load(open('words.pkl', 'rb'))
    labels = pickle.load(open('classes.pkl', 'rb'))
    intents = json.loads(open('intents.json').read())
    model = load_model('chatbot_model.h5')
    print('words == ' + str(words))
    print('labels == ' + str(labels))
    print('intents == ' + str(intents))
    chat(words, labels, intents, model)


if __name__=="__main__":
    main()
