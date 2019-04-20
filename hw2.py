import nltk
from nltk.corpus import treebank
from nltk.tag import tnt
from collections import defaultdict, Counter
import random
import numpy as np
import pandas
import operator

nltk.download('treebank')
# Part 1 (data preparation):
len(treebank.tagged_sents())

# output: 3914

train_data = treebank.tagged_sents()[:3000]
test_data = treebank.tagged_sents()[3000:]
print train_data


# Part 2(Simple tagger)

class simple_tagger:
    def __init__(self):
        self.words_to_tag_freq = defaultdict(int)
        self.words_to_tag_map = defaultdict(str)

    def train(self, data):
        word_count_dict = defaultdict(int)
        for list_of_word_tag_pair in data:
            for word_tag_pair in list_of_word_tag_pair:
                word_to_count = word_tag_pair[0]
                key_to_count = word_tag_pair[1]
                counter_for_key_when_word = 0

                for word_tag_search_pair in list_of_word_tag_pair:
                    word_to_compare = word_tag_search_pair[0]
                    if word_to_count == word_to_compare:
                        key_to_compare = word_tag_search_pair[1]
                        if key_to_count == key_to_compare:
                            counter_for_key_when_word += 1.0
                            list_of_word_tag_pair.remove(word_tag_search_pair)
                self.words_to_tag_freq[word_to_count] = counter_for_key_when_word
                self.words_to_tag_map[word_to_count] = key_to_count


    def evaluate(self, data):
        number_of_successes_word = 0.0
        number_of_successes_sentence = 0.0
        number_of_words = 0.0
        for sentence in data:
            number_of_words_in_sentence = 0.0
            for word_tag_pair in sentence:
                word_to_search = word_tag_pair[0]
                key_to_search = word_tag_pair[1]
                key_to_compare = self.words_to_tag_map.get(word_to_search)
                if key_to_compare is not None:
                    if key_to_search == key_to_compare:
                        number_of_successes_word += 1.0
                        number_of_words_in_sentence += 1.0
                else:
                    self.words_to_tag_map[word_to_search] = random.choice(self.words_to_tag_map.keys())
                number_of_words += 1.0
            if number_of_words_in_sentence == len(sentence):
                number_of_successes_sentence += 1.0

        accuracy_of_sentences = float(number_of_successes_sentence / len(data))
        accuracy_of_words = float(number_of_successes_word / number_of_words)
        return accuracy_of_words, accuracy_of_sentences


# Part 3(HMM tagger)

class hmm_tagger:
    def __init__(self):
        self.matrix_A = None
        self.probs_of_matrix_A = None
        self.matrix_B = None
        self.probs_of_matrix_B = None
        self.matrix_Pi = None
        self.probs_of_matrix_Pi = None

        self.B_list = list()

        self.list_of_words_from_data = list
        self.list_of_tags_from_data = list

    def defin_matrix(self, height, width, rows_names, columns_names):
        matrix = np.zeros(shape=(height, width))
        return pandas.DataFrame(matrix, rows_names, columns_names)

    def train(self, data):
        self.initialize_list_of_words_data_and_tags(data)
        # Initial matrix A
        matrix_a_names = [tag for tag in self.list_of_tags_from_data]
        self.matrix_A = self.defin_matrix(len(self.list_of_tags_from_data), len(self.list_of_tags_from_data),
                                          matrix_a_names,  matrix_a_names)
        # Initial matrix B
        matrix_b_names = [word for word in self.list_of_words_from_data]
        self.matrix_B = self.defin_matrix(len(self.list_of_tags_from_data), len(self.list_of_words_from_data),
                                          matrix_a_names, matrix_b_names)
        # Initial matrix Pi
        self.matrix_Pi = np.zeros(shape=(len(self.list_of_tags_from_data)))
        self.matrix_Pi = pandas.DataFrame(self.matrix_Pi, matrix_a_names)

        for sentence in list(data):
            # Handle first word+tag
            first_pair_word_tag_of_sentence = sentence[0]
            self.matrix_B.loc[first_pair_word_tag_of_sentence[1]][first_pair_word_tag_of_sentence[0]] += 1
            self.matrix_Pi.loc[first_pair_word_tag_of_sentence[1]] += 1
            last_word_tag_pair = first_pair_word_tag_of_sentence
            for word_tag_pair in sentence[1:]:
                self.matrix_A.loc[last_word_tag_pair[1]][word_tag_pair[1]] += 1
                self.matrix_B.loc[word_tag_pair[1]][word_tag_pair[0]] += 1
                self.matrix_Pi.loc[word_tag_pair[1]] += 1

                last_word_tag_pair = word_tag_pair
        total_string_key = "total"
        self.probs_of_matrix_A = self.sum_matrix_columns(total_string_key, self.matrix_A)
        self.probs_of_matrix_B = self.sum_matrix_columns(total_string_key, self.matrix_B)
        self.probs_of_matrix_Pi = self.sum_Pi_to_probs()

    def initialize_list_of_words_data_and_tags(self, data):
        self.list_of_words_from_data = [[word_tag_pair[0] for word_tag_pair in sentence] for sentence in data]
        self.list_of_words_from_data = set(reduce(lambda w1, w2: w1 + w2, self.list_of_words_from_data))
        self.list_of_tags_from_data = [[word_tag_pair[1] for word_tag_pair in sentence] for sentence in data]
        self.list_of_tags_from_data = set(reduce(lambda w1, w2: w1 + w2, self.list_of_tags_from_data))

    def sum_Pi_to_probs(self):
        total_of_Pi = self.matrix_Pi.values.sum()
        return self.matrix_Pi.div(total_of_Pi).fillna(0)

    def sum_matrix_columns(self, total_string_key, matrix):
        matrix[total_string_key] = self.matrix_A.sum(1)
        return self.matrix_A.div(matrix[total_string_key], 0).fillna(0).drop(total_string_key, 1)

    def evaluate(self, data):
        number_of_successes_word = 0.0
        number_of_successes_sentence = 0.0
        number_of_words = 0.0
        unknown_segment_string = "U/K"
        for sentence in data:
            number_of_words += len(sentence)
            list_of_words = [word_tag_pair[0] for word_tag_pair in sentence]
            list_of_tags = [word_tag_pair[1] for word_tag_pair in sentence]
            list_of_indexes = []
            list_of_segments = []
            for word in list_of_words:
                try:
                    list_of_indexes.append(self.probs_of_matrix_B.columns.get_loc(word))
                except KeyError:
                    if len(list_of_indexes) > 0:
                        list_of_segments.append(list_of_indexes)
                    list_of_segments.append(unknown_segment_string)
                    list_of_indexes = []
            if len(list_of_indexes) > 0:
                list_of_segments.append(list_of_indexes)

            list_of_word_sequences = []
            for segment in list_of_segments:
                if segment == unknown_segment_string:
                    list_of_word_sequences.append(unknown_segment_string)
                    continue
                sequence = viterbi(segment, self.probs_of_matrix_A.values, self.probs_of_matrix_B.values,
                                   self.probs_of_matrix_Pi.iloc[:, 0].values)
                list_of_word_sequences.append(sequence)
            list_of_tags_sequences = []
            for sequence in list_of_word_sequences:
                if str(sequence) == unknown_segment_string:
                    new_tag = random.choice(list_of_tags)
                    list_of_tags_sequences.append(new_tag)
                else:
                    for index in sequence:
                        new_tag = self.matrix_A.values[int(index)]
                        list_of_tags_sequences.append(new_tag)

            number_of_words_in_sentence = 0.0
            for indexOfTag, TAG in enumerate(list_of_tags_sequences):
                if str(TAG) == list_of_tags[indexOfTag]:
                    number_of_words_in_sentence += 1.0
                    number_of_successes_word += 1.0
            if number_of_words_in_sentence == len(sentence):
                number_of_successes_sentence += 1.0
        accuracy_of_sentences = number_of_successes_sentence / len(data)
        accuracy_of_words = number_of_successes_word / number_of_words

        return accuracy_of_words, accuracy_of_sentences


def viterbi(word_list, A, B, Pi):
    # initialization
    T = len(word_list)
    N = A.shape[0]  # number of tags

    delta_table = np.zeros((N, T))  # initialise delta table
    psi = np.zeros((N, T))  # initialise the best path table

    delta_table[:, 0] = B[:, word_list[0]] * Pi

    for t in range(1, T):
        for s in range(0, N):
            trans_p = delta_table[:, t - 1] * A[:, s]
            psi[s][t], delta_table[s][t] = max(enumerate(trans_p), key=operator.itemgetter(1))
            delta_table[s][t] = delta_table[s][t] * B[s][word_list[t]]

    # Back tracking
    seq = np.zeros(T)
    seq[T - 1] = delta_table[:, T - 1].argmax()
    for t in range(T - 1, 0, -1):
        seq[t - 1] = psi[int(seq[t])][t]

    return seq


if __name__ == '__main__':
    A = np.array([[0.3, 0.7], [0.2, 0.8]])
    B = np.array([[0.1, 0.1, 0.3, 0.5], [0.3, 0.3, 0.2, 0.2]])
    Pi = np.array([0.4, 0.6])
    print(viterbi([3, 3, 3, 3], A, B, Pi))


# Part 4
simpleTagger = simple_tagger()
simpleTagger.train(train_data)
print simpleTagger.evaluate(test_data)

hmmTagger = hmm_tagger()
hmmTagger.train(train_data)
print hmmTagger.evaluate(test_data)

tnt_pos_tagger = tnt.TnT()
tnt_pos_tagger.train(train_data)

tnt_number_of_successes_word = 0.0
tnt_number_of_successes_sentence = 0.0

tnt_list_of_words_from_data = [[word_tag_pair[0] for word_tag_pair in sentence] for sentence in test_data]
tnt_list_of_all_words_from_data = reduce(lambda w1, w2: w1 + w2, tnt_list_of_words_from_data)
tnt_list_of_tags_from_data = [[word_tag_pair[1] for word_tag_pair in sentence] for sentence in test_data]
for index_of_sentence, sentence in enumerate(tnt_list_of_words_from_data):
    tnt_number_of_words_in_sentence = 0.0
    tnt_list_of_tags = tnt_pos_tagger.tag(sentence)
    for index_of_tag, tag in enumerate(tnt_list_of_tags):
        if tag[1] == tnt_list_of_words_from_data[index_of_sentence][index_of_tag]:
            tnt_number_of_words_in_sentence += 1
            tnt_number_of_successes_word += 1
    if len(tag) == tnt_number_of_words_in_sentence:
        tnt_number_of_successes_sentence += 1
tnt_accuracy_of_sentences = float(tnt_number_of_successes_sentence / len(train_data))
tnt_accuracy_of_words = float(tnt_number_of_successes_word / len(tnt_list_of_all_words_from_data))

print (tnt_accuracy_of_words, tnt_accuracy_of_sentences)

