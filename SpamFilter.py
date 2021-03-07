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


def removeStopwords(text):
    '''Remove all stopwords in a piece of unpunctuated text'''
    text_list = list(text.split(' '))
    clean_text_list = [
        x for x in text_list if x not in stopwords.words('english')
    ]
    
    return clean_text_list



def main():
    text = 'Hi!! I am, a good boy.'
    unpunctuated_text = removePunctuations(text)
    print(f'Unpuntuated Text: {unpunctuated_text}')
    clean_text_list = removeStopwords(unpunctuated_text)
    print(f'Clean Text List: {clean_text_list}')


if __name__ == '__main__':
    main()
