import collections
import re
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
from tqdm import tqdm


def counter_words(sentences):
    word_counter = collections.Counter()
    word_list = []
    num_lines, num_words = (0, 0)
    for i in sentences:
        words = re.findall('[\\w\']+|[;:\-\(\)&.,!?"]', i)
        word_counter.update(words)
        word_list.extend(words)
        num_lines += 1
        num_words += len(words)
    return word_counter, word_list, num_lines, num_words


def build_dict(word_counter, vocab_size = 50000):
    count = [['PAD', 0], ['UNK', 1], ['START', 2], ['END', 3]]
    count.extend(word_counter.most_common(vocab_size))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    return dictionary, {word: idx for idx, word in dictionary.items()}


def doc2num(word_list, dictionary):
    word_array = []
    unknown_val = len(dictionary)
    for word in word_list:
        word_array.append(dictionary.get(word, unknown_val))
    return np.array(word_array, dtype = np.int32)


def build_word_array(sentences, vocab_size):
    word_counter, word_list, num_lines, num_words = counter_words(sentences)
    dictionary, rev_dictionary = build_dict(word_counter, vocab_size)
    word_array = doc2num(word_list, dictionary)
    return word_array, dictionary, rev_dictionary, num_lines, num_words


def build_training_set(word_array):
    num_words = len(word_array)
    x = np.zeros((num_words - 4, 4), dtype = np.int32)
    y = np.zeros((num_words - 4, 1), dtype = np.int32)
    shift = np.array([-2, -1, 1, 2], dtype = np.int32)
    for idx in range(2, num_words - 2):
        y[idx - 2, 0] = word_array[idx]
        x[idx - 2, :] = word_array[idx + shift]
    return x, y


class Model:
    def __init__(self, graph_params):
        g_params = graph_params
        tf.compat.v1.reset_default_graph()
        self.sess = tf.compat.v1.InteractiveSession()
        self.X = tf.compat.v1.placeholder(tf.compat.v1.int64, shape = [None, 4])
        self.Y = tf.compat.v1.placeholder(tf.compat.v1.int64, shape = [None, 1])
        w_m2, w_m1, w_p1, w_p2 = tf.compat.v1.unstack(self.X, axis = 1)
        self.embed_weights = tf.compat.v1.Variable(
            tf.compat.v1.random_uniform(
                [g_params['vocab_size'], g_params['embed_size']],
                -g_params['embed_noise'],
                g_params['embed_noise'],
            )
        )
        embed_m2 = tf.compat.v1.nn.embedding_lookup(self.embed_weights, w_m2)
        embed_m1 = tf.compat.v1.nn.embedding_lookup(self.embed_weights, w_m1)
        embed_p1 = tf.compat.v1.nn.embedding_lookup(self.embed_weights, w_p1)
        embed_p2 = tf.compat.v1.nn.embedding_lookup(self.embed_weights, w_p2)
        embed_stack = tf.compat.v1.concat([embed_m2, embed_m1, embed_p1, embed_p2], 1)
        hid_weights = tf.compat.v1.Variable(
            tf.compat.v1.random_normal(
                [g_params['embed_size'] * 4, g_params['hid_size']],
                stddev = g_params['hid_noise']
                / (g_params['embed_size'] * 4) ** 0.5,
            )
        )
        hid_bias = tf.compat.v1.Variable(tf.compat.v1.zeros([g_params['hid_size']]))
        hid_out = tf.compat.v1.nn.tanh(tf.compat.v1.matmul(embed_stack, hid_weights) + hid_bias)
        self.nce_weights = tf.compat.v1.Variable(
            tf.compat.v1.random_normal(
                [g_params['vocab_size'], g_params['hid_size']],
                stddev = 1.0 / g_params['hid_size'] ** 0.5,
            )
        )
        nce_bias = tf.compat.v1.Variable(tf.compat.v1.zeros([g_params['vocab_size']]))
        self.cost = tf.compat.v1.reduce_mean(
            tf.compat.v1.nn.nce_loss(
                self.nce_weights,
                nce_bias,
                inputs = hid_out,
                labels = self.Y,
                num_sampled = g_params['neg_samples'],
                num_classes = g_params['vocab_size'],
                num_true = 1,
                remove_accidental_hits = True,
            )
        )
        self.logits = tf.compat.v1.argmax(
            tf.compat.v1.matmul(hid_out, self.nce_weights, transpose_b = True) + nce_bias,
            axis = 1,
        )
        if g_params['optimizer'] == 'RMSProp':
            self.optimizer = tf.compat.v1.train.RMSPropOptimizer(
                g_params['learn_rate']
            ).minimize(self.cost)
        elif g_params['optimizer'] == 'Momentum':
            self.optimizer = tf.compat.v1.train.MomentumOptimizer(
                g_params['learn_rate'], g_params['momentum']
            ).minimize(self.cost)
        elif g_params['optimizer'] == 'Adam':
            self.optimizer = tf.compat.v1.train.AdamOptimizer(
                g_params['learn_rate']
            ).minimize(self.cost)
        else:
            print('Optimizer not supported,exit.')
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def train(self, X, Y, X_val, Y_val, epoch, batch_size):
        for i in range(epoch):
            X, Y = shuffle(X, Y)
            pbar = tqdm(
                range(0, len(X), batch_size), desc = 'train minibatch loop'
            )
            for batch in pbar:
                feed_dict = {
                    self.X: X[batch : min(batch + batch_size, len(X))],
                    self.Y: Y[batch : min(batch + batch_size, len(X))],
                }
                _, loss = self.sess.run(
                    [self.optimizer, self.cost], feed_dict = feed_dict
                )
                pbar.set_postfix(cost = loss)

            pbar = tqdm(
                range(0, len(X_val), batch_size), desc = 'test minibatch loop'
            )
            for batch in pbar:
                feed_dict = {
                    self.X: X_val[batch : min(batch + batch_size, len(X_val))],
                    self.Y: Y_val[batch : min(batch + batch_size, len(X_val))],
                }
                loss = self.sess.run(self.cost, feed_dict = feed_dict)
                pbar.set_postfix(cost = loss)
        return self.embed_weights.eval(), self.nce_weights.eval()


class Word2Vec:
    def __init__(self, embed_matrix, dictionary):
        self._embed_matrix = embed_matrix
        self._dictionary = dictionary
        self._reverse_dictionary = {v: k for k, v in dictionary.items()}

    def get_vector_by_name(self, word):
        return np.ravel(self._embed_matrix[self._dictionary[word], :])

    def n_closest(self, word, num_closest = 5, metric = 'cosine'):
        wv = self.get_vector_by_name(word)
        closest_indices = self.closest_row_indices(wv, num_closest + 1, metric)
        word_list = []
        for i in closest_indices:
            word_list.append(self._reverse_dictionary[i])
        if word in word_list:
            word_list.remove(word)
        return word_list

    def closest_row_indices(self, wv, num, metric):
        dist_array = np.ravel(
            cdist(self._embed_matrix, wv.reshape((1, -1)), metric = metric)
        )
        sorted_indices = np.argsort(dist_array)
        return sorted_indices[:num]

    def analogy(self, a, b, c, num = 1, metric = 'cosine'):
        va = self.get_vector_by_name(a)
        vb = self.get_vector_by_name(b)
        vc = self.get_vector_by_name(c)
        vd = vb - va + vc
        closest_indices = self.closest_row_indices(vd, num, metric)
        d_word_list = []
        for i in closest_indices:
            d_word_list.append(self._reverse_dictionary[i])
        return d_word_list

    def project_2d(self, start, end):
        tsne = TSNE(n_components = 2)
        embed_2d = tsne.fit_transform(self._embed_matrix[start:end, :])
        word_list = []
        for i in range(start, end):
            word_list.append(self._reverse_dictionary[i])
        return embed_2d, word_list
