import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

book_names = ["Frankenstein.txt", "Odyssey.txt", "Romeo_and_Juliet.txt"]

corpus = []

stop_words = {"the", "it", "this", "or", "so", "me", "person", "than", "back", "even", "be", "for", "but", "will", "up",
              "make", "into", "then", "after", "new", "to", "not", "his", "an", "out", "can", "year", "now", "use",
              "want", "of", "on", "by", "my", "if", "like", "your", "look", "two", "because", "and", "with", "from",
              "one", "about", "time", "good", "only", "how", "any", "a", "he", "they", "all", "who", "no", "some",
              "come", "our", "these", "in", "as", "we", "would", "get", "just", "could", "its", "work", "give", "that",
              "you", "say", "there", "which", "him", "them", "over", "first", "day", "have", "do", "her", "their", "go",
              "know", "see", "think", "well", "most", "I", "at", "she", "what", "when", "take", "other", "also", "way",
              "us", "has"}

words_set = set()

for doc in book_names:
    with open(doc, "r") as file:
        data = file.read()
    file.close()

    cleaned_data = re.sub(r'[^\w\s]', "", data.lower())
    cleaned_data = re.sub(r'[\n_]', " ", cleaned_data)

    # all_words_ind = cleaned_data.split()
    corpus.append(cleaned_data)

for i in range(len(corpus)):
    all_words_ind = corpus[i].split()
    all_words_ind = [str(word) for word in all_words_ind if word not in stop_words]
    words_set = words_set.union(set(all_words_ind))

n_words_set = len(words_set)

df_tf = pd.DataFrame(np.zeros((len(corpus), n_words_set)), columns=list(words_set))

# Compute the TF for TF-IDF

for i in range(len(corpus)):
    words = corpus[i].split()
    total_words = len(words)
    for word in words:
        if word in words_set:
            df_tf.loc[i][word] += words.count(word) / total_words

idf_dict = {}

for word in words_set:
    num_containing = 0

    for i in range(len(corpus)):
        if word in corpus[i].split():
            num_containing += 1

    idf_dict[word] = np.log10(len(corpus) / num_containing)

df_tf_idf = df_tf.copy()

for word in words_set:
    for i in range(len(corpus)):
        df_tf_idf.loc[i, word] = df_tf.loc[i, word] * idf_dict[word]

top_tf_idf_per_novel = []

for i in range(len(corpus)):
    top_words = df_tf_idf.loc[i].nlargest(5)
    top_tf_idf_per_novel.append(top_words)

for i, book_name in enumerate(book_names):
    print(f"Top 5 IF-IDF Scores in '{book_name}':")
    print(top_tf_idf_per_novel[i])

fig, axs = plt.subplots(1, 3, figsize=(15,3))

for i in range(len(corpus)):
    book_name = book_names[i]
    top_words = top_tf_idf_per_novel[i]
    x = top_words.index

    axs[i].bar(x, top_words.values, color="skyblue")
    axs[i].set_xlabel(f"Top 5 TF-IDF scores in '{book_name}'")
    axs[i].set_ylabel("TF-IDF Score")

plt.tight_layout()
plt.show()
