from googletrans import Translator

trans = Translator(service_urls=['translate.googleapis.com'])

def translate(input_texts=[]):
    global trans
    output_texts=[]
    for text in input_texts:
        output_texts.append(trans.translate(text, src='en', dest='ru').text)
    return output_texts