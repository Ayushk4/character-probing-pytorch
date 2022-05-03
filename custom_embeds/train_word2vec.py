import gensim
from gensim.models.word2vec import LineSentence
import pickle as pkl
import os, sys

try:
    os.mkdir("vecs/")
except:
    pass

lines = LineSentence(sys.argv[1])

print("Prepared Data")
print("Training Word2Vec")
model = gensim.models.Word2Vec(lines, vector_size=100,
                window=5, min_count=5, workers=10, epochs=5)
print("Trained Word2Vec")
model.save("vecs/" + sys.argv[1].split("/")[-1] + ".vec")

