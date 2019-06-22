import numpy
import numpy as np
import itertools
from collections import deque
from gensim.models import Word2Vec
from sklearn import svm


def conll_to_transitions(sentence):
    '''
    Given a sentence, returns a list of transitions.
    Each transition is a training instance for your classifier. A transition
    is composed of the following 4 items:
    - first word in stack
    - second word in stack (could be None is stack is of size=1)
    - first word in buffer (could be None if the buffer is empty)
    - the transition label (SHIFT, LEFT, RIGHT)
    '''
    s = []  # stack
    b = deque([])  # buffer

    transitions = []

    for w in sentence:
        b.append(w)

    s.append(['0', 'ROOT', '_', '_', '_', '_', '0', '_', '_', '_'])

    while len(b) > 0 or len(s) > 1:
        if s[-1][0] == '0':  # the root
            add_shift(s, b, transitions)
        elif s[-2][6] == s[-1][0] and check_rightest_arc(s[-2], b):
            add_left(s, b, transitions)
        elif s[-1][6] == s[-2][0] and (len(b) == 0 or s[-2][0] != '0') and check_rightest_arc(s[-1], b):
            add_right(s, b, transitions)
        elif len(b) == 0:
            return None
        else:
            add_shift(s, b, transitions)
    return transitions


def check_rightest_arc(word, b):
    '''
   w[6] is the index of the head of "this" word, so in this method we check
   if there is an arc that goes from one of the words in the buffer
   to "word" (which exists in the stack)
    '''
    for w in b:
        if w[6] == word[0]:
            return False
    return True


def add_shift(s, b, transitions):
    '''
    Adding shift transition
    '''
    word = b.popleft()
    top2 = None
    if len(s) > 1:
        top2 = s[-2]
    transitions.append([s[-1], top2, word, 'SHIFT'])
    s.append(word)


def add_left(s, b, transitions):
    '''
    Adding left transition
    '''
    top1 = s.pop()
    top2 = s.pop()
    transitions.append([top1, top2, b[0] if len(b) > 0 else None, 'LEFT'])
    s.append(top1)


def add_right(s, b, transitions):
    '''
    Adding right transition
    '''
    top1 = s.pop()
    top2 = s.pop()
    transitions.append([top1, top2, b[0] if len(b) > 0 else None, 'RIGHT'])
    s.append(top2)


def one_hot_array_for_pos(splitted_line_from_file, dictionary_of_pos_to_index):
    length = len(dictionary_of_pos_to_index)
    array = numpy.zeros(length)
    if splitted_line_from_file is None:
        temp_value = "UNK"
    else:
        temp_value = splitted_line_from_file[3]
    if dictionary_of_pos_to_index.__contains__(temp_value):
        array[dictionary_of_pos_to_index[temp_value]] = 1
    else:
        temp_value = "UNK"
        array[dictionary_of_pos_to_index[temp_value]] = 1

    return array


def embedded_word_by_model(model, splitted_line_from_file):
    try:
        embedded_word = model[splitted_line_from_file[1]]
        return embedded_word
    except:
        return numpy.zeros(model.layer1_size)


def create_vector(model, list_of_transitions, pos):
    transition_stack_1 = list_of_transitions[0]
    transition_stack_2 = list_of_transitions[1]
    transition_buffer = list_of_transitions[2]

    embeded_transition_stack_1 = embedded_word_by_model(model, transition_stack_1)
    embeded_transition_stack_2 = embedded_word_by_model(model, transition_stack_2)
    embeded_transition_buffer = embedded_word_by_model(model, transition_buffer)

    one_hot_array_transition_stack_1 = one_hot_array_for_pos(transition_stack_1, pos)
    one_hot_array_transition_stack_2 = one_hot_array_for_pos(transition_stack_2, pos)
    one_hot_array_transition_buffer = one_hot_array_for_pos(transition_buffer, pos)
    return list(itertools.chain(embeded_transition_stack_1, embeded_transition_stack_2, embeded_transition_buffer,
                                one_hot_array_transition_stack_1, one_hot_array_transition_stack_2,
                                one_hot_array_transition_buffer))


def create_transition_list(buffer_of_lines, stack_of_lines):
    if len(stack_of_lines) >= 2:
        word_to_chain_2 = stack_of_lines[-2]
    else:
        word_to_chain_2 = None

    if len(stack_of_lines) >= 1:
        word_to_chain_1 = stack_of_lines[-1]
    else:
        word_to_chain_1 = None

    if len(buffer_of_lines) >= 1:
        word_to_chain_3 = buffer_of_lines[-1]
    else:
        word_to_chain_3 = None

    return [word_to_chain_1, word_to_chain_2, word_to_chain_3]


def parse(model, sentence_to_parse, pos, model_word2vec):
    buffer_of_lines, left_string, list_of_possible_transitions, list_of_transitions, right_string, shift_string, stack_of_lines = inital_parsing_vars(
        sentence_to_parse)

    try:
        while stack_of_lines:
            transitions_list = create_transition_list(buffer_of_lines, stack_of_lines)
            vector = create_vector(model_word2vec, transitions_list, pos)
            decision = model.decision_function([vector])
            transitions_chosen = list_of_possible_transitions[np.argmax(decision)]

            handle_transitions(buffer_of_lines, left_string, list_of_transitions, right_string, shift_string,
                               stack_of_lines, transitions_chosen)
    except:
        pass
    return list_of_transitions


def inital_parsing_vars(sentence_to_parse):
    list_of_transitions = []
    stack_of_lines = []
    buffer_of_lines = []
    left_string = "LEFT"
    right_string = "RIGHT"
    shift_string = "SHIFT"
    # from kfir code :
    stack_of_lines.append(['0', 'ROOT', '_', '_', '_', '_', '0', '_', '_', '_'])
    list_of_possible_transitions = {0: shift_string, 1: right_string, 2: left_string}
    for index in range(len(sentence_to_parse) - 1, 0, -1):
        buffer_of_lines.append(sentence_to_parse[1])
    return buffer_of_lines, left_string, list_of_possible_transitions, list_of_transitions, right_string, shift_string, stack_of_lines


def handle_transitions(buffer_of_lines, left_string, list_of_transitions, right_string, shift_string, stack_of_lines,
                       transitions_chosen):
    if transitions_chosen == shift_string:
        list_of_transitions.append(shift_string)
        stack_of_lines.append(buffer_of_lines.pop())
    if transitions_chosen == right_string:
        list_of_transitions.append(right_string)
        buffer_of_lines.pop()
    if transitions_chosen == left_string:
        list_of_transitions.append(left_string)
        top_sent = stack_of_lines.pop()
        stack_of_lines.pop()
        stack_of_lines.append(top_sent)


def train():
    global list_of_training_data, list_of_output_data, f, line, dictionary_of_pos_to_index, model_from_sentences, clf
    list_of_sentences = []
    list_of_sentences_learning = []
    list_of_current_sentence = []
    list_of_current_sentence_learning = []
    list_of_transitions = []
    left_string = "LEFT"
    right_string = "RIGHT"
    shift_string = "SHIFT"
    unknown_string = "UNK"
    set_of_part_of_speech = set()
    set_of_part_of_speech.add(unknown_string)
    enum_transitions = {shift_string: 0, right_string: 1, left_string: 2}
    list_of_training_data = []
    list_of_output_data = []
    with open('train', 'r') as f:
        for line in f:
            if not line.strip():
                list_of_sentences_learning.append(list_of_current_sentence_learning)
                list_of_current_sentence_learning = []
                list_of_sentences.append(list_of_current_sentence)
                list_of_current_sentence = []
                continue
            list_of_words = line.split()
            index_of_word_in_sentence = list_of_words[0]
            the_word_itself = list_of_words[1]
            POS_tag_coarse = list_of_words[3]
            POS_tag_more_specific = list_of_words[4]
            head_index_of_words = list_of_words[6]
            dependecy_label = list_of_words[7]

            list_of_current_sentence.append(the_word_itself)
            list_of_current_sentence_learning.append(list_of_words)
            if not POS_tag_coarse in set_of_part_of_speech:
                set_of_part_of_speech.add(POS_tag_coarse)
            if not POS_tag_more_specific in set_of_part_of_speech:
                set_of_part_of_speech.add(POS_tag_more_specific)
    dictionary_of_pos_to_index = dict(
        {(part_of_speech, index) for index, part_of_speech in enumerate(set_of_part_of_speech)})
    for index_of_sentence in range(len(list_of_sentences_learning)):
        sent = list_of_sentences_learning[index_of_sentence]
        list_of_transitions.append(conll_to_transitions(sent))
    model_from_sentences = Word2Vec(list_of_sentences, min_count=1)
    for transitions in list_of_transitions:
        if not transitions is None:
            for transition in transitions:
                list_of_training_data.append(
                    create_vector(model_from_sentences, transition, dictionary_of_pos_to_index))
                list_of_output_data.append(enum_transitions[transition[3]])
    clf = svm.SVC(gamma='scale', decision_function_shape='ovr')
    clf.fit(list_of_training_data, list_of_output_data)


def eval():
    global counter_for_current_success, counter_for_total_success, f, line
    counter_for_current_success = 0.0
    counter_for_total_success = 0.0
    with open('eval', 'r') as f:
        list_of_splitted_line = []
        for line in f.readlines():
            if not line == '\n':
                list_of_splitted_line.append(line.split())
            else:
                sentence_after_parsing = conll_to_transitions(list_of_splitted_line)
                sentence_parsed = parse(clf, list_of_splitted_line, dictionary_of_pos_to_index, model_from_sentences)
                list_of_splitted_line = []
                if not sentence_after_parsing is None:
                    counter_for_total_success += len(sentence_after_parsing)
                    for i in range(min(len(sentence_after_parsing), len(sentence_parsed))):
                        if sentence_parsed[i] == sentence_after_parsing[i][3]:
                            counter_for_current_success += 1.0


train()
eval()
print(float(float(counter_for_current_success) / float(counter_for_total_success)))
