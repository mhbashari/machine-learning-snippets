import os

from hazm import word_tokenize, Normalizer


class Global:
    normal = Normalizer()


def sentence_line_doc_reader(file_path):
    for line in open(file_path, "r", encoding="utf-8"):
        yield word_tokenize(Global.normal.normalize(line.replace("\n", "")))


def sentence_line_dir_reader(directory):
    for f in os.listdir(directory):
        for sentence in sentence_line_doc_reader(os.path.join(directory, f)):
            yield sentence
