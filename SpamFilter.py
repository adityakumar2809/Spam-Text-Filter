import string
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords


def removePunctuations(text):
    '''Remove all punctuations from a piece of text'''
    unpunctuated_text = ''
    for ch in text:
        if ch not in string.punctuation:
            unpunctuated_text += ch

    return unpunctuated_text


def main():
    text = 'Hi!! I am, a good boy.'
    unpunctuated_text = removePunctuations(text)


if __name__ == '__main__':
    main()
