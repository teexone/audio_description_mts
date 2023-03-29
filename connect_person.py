import numpy as np

from face.face_recognition import recognize_faces
from nltk.corpus import wordnet as wn

def detect_people(caption):
    best_word, best_score = None, 0
    for word in caption:
        word_sysnet = wn.synsets(word)
        for ws in word_sysnet:
            lch = ws.lowest_common_hypernyms(wn.synset("person.n.01"))
            if wn.synset("person.n.01") in lch:
                score = wn.path_similarity(wn.synset("person.n.01"), ws)
                if score > best_score:
                    best_score = score
                    best_word = word

    return best_word


def best_alpha(alpha, bxs):
    index, best = -1, 0.
    for i, bx in enumerate(bxs):
        s = alpha.view(16, 16)[bx[0]:bx[2], bx[1]:bx[3]].sum()
        if s > best:
            index = i
            best = s
    return index, best

def connect_person_no_attention(
        image_paths,
        captions,
        embeddings,
):
    modified = []
    for ipath, caption in zip(image_paths, captions):
        recognized = recognize_faces(ipath, embeddings, list(embeddings.keys()))
        if len(recognized) != 1:
            modified.append(caption)
            continue
        else:
            splitted = caption.split(" ")
            person = detect_people(splitted)
            if person is None:
                modified.append(caption)
                continue
            splitted[splitted.index(person)] = recognized[0][0]
            modified.append(" ".join(splitted))
    return modified

