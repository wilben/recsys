# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 13:07:07 2017

@author: Administrator
"""



import time
import importlib
import re
import numpy as np
import tensorflow as tf
import tensorlayer as tl
import json
import collections
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math


class ReviewRatingData:
    """use to save raviews and ratings"""

    def __init__(self, filename, n_maxwords=50000, n_minwordfreq=3):
        # f = open("Amazon_Instant_Video_5.json", encoding='utf-8')

        # jsondata=[]#the list of dict of every rating&review
        # self.data=[]#onehot representation of jsondata:users,items,reviews,ratings
        words = []  # a list of all words
        reviews = []  # a list of list of all words in a review
        reviewswordscount = []
        users = []  # a list of all users in reviews
        items = []
        ratings = []

        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                # temp = json.loads(f.read())
                jsondata = (json.loads(line))
                # for i in range(len(self.jsondata)):
                reviews.append([w for w in re.split(r'["/-`()!?,;.\s]', jsondata['reviewText'].lower()) if w])
                reviewswordscount.append(len(reviews[i]))  # the last reviews' words count

                # assert i==len(reviews)-1
                words.extend(reviews[i])
                users.append(jsondata['reviewerID'])
                items.append(jsondata['asin'])
                ratings.append(int(jsondata['overall']))

        assert len(users) == len(items) == len(reviews) == len(ratings)

        self.totalReviewNum = len(users)

        self.__codingData(words, reviews, reviewswordscount, users, items, ratings, n_maxwords, n_minwordfreq)
        # it's not necessary that the users/items in the Test Set must have been showed up in the train set.
        # shuffle the data,then build sets
        self.shuffleTrain_CV_TestSet()


    '''
    def getReviewsByUV(self,u,v):
        return self.onehot_reivews[u][v]
    '''

    # def getData(self,i,field):
    #    return self.onehot_jsondata[field][i]
    def shuffleTrain_CV_TestSet(self, rate=[0.6, 0.2, 0.2]):
        self.trainReviewNum = int(self.totalReviewNum * rate[0])
        self.CVReviewNum = int(self.totalReviewNum * rate[1])
        self.testReviewNum = self.totalReviewNum - self.trainReviewNum - self.CVReviewNum

        self.shuffle_idx = np.random.permutation(self.totalReviewNum)
        print('总共:训练集:验证集:测试集=', self.totalReviewNum, self.trainReviewNum, self.CVReviewNum, self.testReviewNum)

        decoratedU = list(zip(self.shuffle_idx, self.onehot_jsondata['user']))
        decoratedI = list(zip(self.shuffle_idx, self.onehot_jsondata['item']))
        decoratedRe = list(zip(self.shuffle_idx, self.onehot_jsondata['review']))
        decoratedRa = list(zip(self.shuffle_idx, self.onehot_jsondata['rating']))

        decoratedU.sort()
        decoratedI.sort()
        decoratedRe.sort()
        decoratedRa.sort()

        shuffleU = [u for (i, u) in decoratedU]
        shuffleI = [u for (i, u) in decoratedI]
        shuffleRe = [u for (i, u) in decoratedRe]
        shuffleRa = [u for (i, u) in decoratedRa]

        n1 = self.trainReviewNum
        n2 = self.CVReviewNum + n1

        self.traindata = {'user': shuffleU[:n1],
                          'item': shuffleI[:n1],
                          'review': shuffleRe[:n1],
                          'rating': shuffleRa[:n1]}

        self.CVdata = {'user': shuffleU[n1:n2],
                       'item': shuffleI[n1:n2],
                       'review': shuffleRe[n1:n2],
                       'rating': shuffleRa[n1:n2]}

        self.testdata = {'user': shuffleU[n2:],
                         'item': shuffleI[n2:],
                         'review': shuffleRe[n2:],
                         'rating': shuffleRa[n2:]}

        self.calBias = False

    # =============================================================================
    #     #省空间，费时间
    #     def getTrainData(self,field,i):
    #         assert i<self.trainReviewNum
    #         return self.onehot_jsondata[field][self.shuffle_idx[i]]
    #
    #     def getCVData(self,field,i):
    #         assert i<self.CVReviewNum
    #         return self.onehot_jsondata[field][self.shuffle_idx[i+self.trainReviewNum]]
    #
    #     def getTestData(self,field,i):
    #         assert i<self.testReviewNum
    #         return self.onehot_jsondata[field][self.shuffle_idx[i+self.trainReviewNum+self.CVReviewNum]]
    # =============================================================================

    def getTrainData(self, field):
        return self.traindata[field]

    def getCVData(self, field):
        return self.CVdata[field]

    def getTestData(self, field):
        return self.testdata[field]

    def buildCount(data):
        count = []
        count.extend(collections.Counter(data).most_common(None))
        countdic = {key: item for (key, item) in count}
        return countdic

    def __codingData(self, words, reviews, reviewswordscount, users, items, ratings, n_maxwords, n_minwordfreq):
。
        # onehot representation of data:
        wordsdata, self.totalWordDict_countDict, self.onehot_wordsdict, self.reverse_wordsdict \
            = ReviewRatingData.build_onehot(words, n_maxwords, n_minwordfreq)

        usersdata, self.totalUserDict_countDict, self.onehot_usersdict, self.reverse_usersdict \
            = ReviewRatingData.build_onehot(users)

        itemsdata, self.totalItemDict_countDict, self.onehot_itemsdict, self.reverse_itemsdict \
            = ReviewRatingData.build_onehot(items)

        assert len(self.totalItemDict_countDict) == len(self.onehot_itemsdict) == len(self.reverse_itemsdict)
        assert len(self.totalUserDict_countDict) == len(self.onehot_usersdict) == len(self.reverse_usersdict)
        assert len(wordsdata) == sum(reviewswordscount)
        self.userNum = len(self.totalUserDict_countDict)
        self.itemNum = len(self.totalItemDict_countDict)
        self.vocabularyNum = len(self.totalWordDict_countDict)
        self.totalWordNum = len(wordsdata)

        # print('Most common words (+UNK)', self.totalWordDict_countDict)
        print('Sample data', wordsdata[:3], [self.reverse_wordsdict[i] for i in wordsdata[:10]])
        # print('Most common users (+UNK)', self.totalUserDict_countDict)
        print('Sample data', self.reverse_usersdict[3])

        # reviews and ratings  are sparse matrix ,use coordinate list to save them
        # maybe these data will never be used ?
        # self.onehot_reivews=[[[] for y in range(len(self.itemsdict_count)) ] for x in range(len(self.usersdict_count)) ]

        # turn wordsdata to reviewsdata
        c = 0
        reviewsdata = []
        for i in range(len(reviews)):
            # u=self.onehot_usersdict[users[i]]
            # v=self.onehot_itemsdict[items[i]]
            reviewsdata.append(wordsdata[c:c + reviewswordscount[i]])
            # self.onehot_reivews[u][v].extend(reviewsdata[i])
            c = c + reviewswordscount[i]

        assert len(usersdata) == len(itemsdata) == len(reviewsdata) == len(ratings) == self.totalReviewNum
        self.onehot_jsondata = {'user': usersdata, 'item': itemsdata, 'review': reviewsdata, 'rating': ratings}

    def getTrainMeanRating(self):

        user_mean_rating, ucount = [0 for i in range(self.getUserNum())], [0 for i in range(self.getUserNum())]
        item_mean_rating, vcount = [0 for i in range(self.getItemNum())], [0 for i in range(self.getItemNum())]
        # 一款商品中，有哪些用户评价它了？得到这个数组。
        TrainItem_UserArray = [[] for i in range(self.getItemNum())]
        item_mean_term2 = [0 for i in range(self.getItemNum())]

        total_mean_rating = 0

        for (u, v, r) in zip(self.getTrainData('user'), self.getTrainData('item'), self.getTrainData('rating')):
            user_mean_rating[u] += r
            item_mean_rating[v] += r
            total_mean_rating += r
            ucount[u] += 1
            vcount[v] += 1
            TrainItem_UserArray[v].append(u)

        total_mean_rating /= self.getTrainNum()

        for u in range(data.getUserNum()):
            # user_mean_rating[u]/=  self.userDict_countDict[self.reverse_usersdict[u]]
            if ucount[u] == 0:
                print(u, '用户没出现在训练集中')
                user_mean_rating[u] = total_mean_rating
            else:
                user_mean_rating[u] /= ucount[u]

        for v in range(data.getItemNum()):
            # item_mean_rating[v]/= self.itemDict_countDict[self.reverse_itemsdict[v]]


            if vcount[v] == 0:
                print(v, '商品没出现在训练集中')
                item_mean_rating[v] = total_mean_rating
                item_mean_term2[v] = total_mean_rating
            else:
                item_mean_rating[v] /= vcount[v]
                item_mean_term2[v] = sum([user_mean_rating[i] for i in TrainItem_UserArray[v]])
                item_mean_term2[v] /= vcount[v]

        return user_mean_rating, item_mean_rating, total_mean_rating, item_mean_term2

    def getTrainBias(self):

        if self.calBias == False:
            user_mean_rating, item_mean_rating, mean_rating, item_mean_term2 = self.getTrainMeanRating()
            user_bias_with_mean = [i - mean_rating for i in user_mean_rating]
            item_bias_with_mean_and_user = [i - j for (i, j) in zip(item_mean_rating, item_mean_term2)]
            #            user_bias_with_mean = np.array(user_bias_with_mean)[:,None]
            #            item_bias_with_mean_and_user = np.array(item_bias_with_mean_and_user)[None,:]
            self.mean_rating = mean_rating
            self.user_bias_with_mean = user_bias_with_mean
            self.item_bias_with_mean_and_user = item_bias_with_mean_and_user
            self.calBias = True

        return self.mean_rating, self.user_bias_with_mean, self.item_bias_with_mean_and_user

    def getCVBiasListForPredict(self):
        m, u, v = self.getTrainBias()

        CVmean = [m for i in range(self.getCVNum())]
        CVuser_bias_with_mean = [u[i] for i in self.getCVData('user')]
        CVitem_bias_with_mean_and_user = [v[i] for i in self.getCVData('item')]

        return CVmean, CVuser_bias_with_mean, CVitem_bias_with_mean_and_user

    def build_onehot(words, maxcount=None, minfreq=None):
        """Process raw inputs into a dataset."""
        if maxcount != None:
            maxcount -= 1

        count = [['UNK', -1]]
        count.extend(collections.Counter(words).most_common(maxcount))
        # delete a word if it's freq<minfreq
        if minfreq != None:
            assert minfreq >= 1
            for i in range(1, len(count)):
                if count[i][1] < minfreq:
                    count = count[:i]
                    break

        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        data = list()
        unk_count = 0

        for word in words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count += 1
            data.append(index)
        count[0][1] = unk_count

        # be caution,usersdata don't has No.0 at index,but the dict has No.0 user.so we del it
        # if the count of key UNK is less than minfreq,del it
        if minfreq == None:
            minfreq = 1
        if count[0][1] < minfreq:
            count = count[1:]
            data = [data[i] - 1 for i in range(len(data))]
            dictionary = {k: dictionary[k] - 1 for k in dictionary.keys()}
            del dictionary['UNK']

        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

        countdic = {key: item for (key, item) in count}

        return data, countdic, dictionary, reversed_dictionary

        # def getCountDict(self,field):
        # return self.usersdict_count

    # def getDict(self,field)

    def getData(self, field):
        return self.onehot_jsondata[field]

    def getReviewNum(self):
        return self.totalReviewNum

    def getTrainNum(self):
        return self.trainReviewNum

    def getCVNum(self):
        return self.CVReviewNum

    def getTestNum(self):
        return self.testReviewNum

    def getUserNum(self):
        return self.userNum

    def getItemNum(self):
        return self.itemNum

    def gettotalWordNum(self):
        return self.totalWordNum

    def getVocabularyNum(self):
        return self.vocabularyNum

    def generate_batch_NOreviewdata(self,batch_size):

        users = self.getTrainData('user')
        items = self.getTrainData('item')
        ratings = self.getTrainData('rating')
        assert len(users) == len(items) == len(ratings)
        sampleNum = len(users)

        batch_ratings = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        batch_users = np.ndarray(shape=(batch_size), dtype=np.int32)
        batch_items = np.ndarray(shape=(batch_size), dtype=np.int32)
        user_bias_with_mean_batch = np.ndarray(shape=(batch_size), dtype=np.float32)
        item_bias_with_mean_and_user_batch = np.ndarray(shape=(batch_size), dtype=np.float32)
        _, user_bias_with_mean, item_bias_with_mean_and_user = self.getTrainBias()

        sample_idx = 0
        z = 0
        while z == 0:  # 总样本数/BATCHSIZE的余数（最后一段）被丢掉了。
            shuffle_idx = np.random.permutation(sampleNum)
            z += 1
            for d in shuffle_idx:

                user = users[d]
                item = items[d]
                rating = ratings[d]

                batch_users[sample_idx] = user
                batch_items[sample_idx] = item
                batch_ratings[sample_idx] = rating
                user_bias_with_mean_batch[sample_idx] = user_bias_with_mean[user]
                item_bias_with_mean_and_user_batch[sample_idx] = item_bias_with_mean_and_user[item]

                if sample_idx == batch_size - 1:
                    sample_idx = 0

                    yield batch_users, \
                          batch_items, \
                          batch_ratings, \
                          user_bias_with_mean_batch, \
                          item_bias_with_mean_and_user_batch

                else:
                    sample_idx += 1

    def generate_batch(self, batch_size, context_window):
        # global data_idx,data_len
        docs = self.getTrainData('review')
        users = self.getTrainData('user')
        items = self.getTrainData('item')
        ratings = self.getTrainData('rating')

        assert len(docs) == len(users) == len(items) == len(ratings)
        sampleNum = len(docs)
        batch_word_data = np.ndarray(shape=(batch_size * context_window), dtype=np.int32)
        batch_labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

        batch_doc = np.ndarray(shape=(batch_size), dtype=np.int32)
        batch_users = np.ndarray(shape=(batch_size), dtype=np.int32)
        batch_items = np.ndarray(shape=(batch_size), dtype=np.int32)
        batch_ratings = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

        batch_inverse_weight = np.ndarray(shape=(batch_size),dtype=np.int32)

        user_bias_with_mean_batch = np.ndarray(shape=(batch_size), dtype=np.float32)
        item_bias_with_mean_and_user_batch = np.ndarray(shape=(batch_size), dtype=np.float32)

        _, user_bias_with_mean, item_bias_with_mean_and_user = self.getTrainBias()

        assert context_window % 2 == 0
        span = context_window // 2
        # buffer = collections.deque(maxlen=span)
        '''
        if data_index + span >= wordslen:
    
            overlay = batch_size - (data_len-data_idx)
    
            shuffle_idx = np.random.permutation(len(docs))
            batch_labels = np.vstack([labels[data_idx:data_len],labels[:overlay]])
            batch_doc_data = np.vstack([doc[data_idx:data_len],doc[:overlay]])
            batch_word_data = np.vstack([context[data_idx:data_len],context[:overlay]])
            data_idx = overlay
        '''

        sample_idx = 0
        z = 0
        while z == 0:  # 总样本数/BATCHSIZE的余数（最后一段）被丢掉了。
            shuffle_idx = np.random.permutation(sampleNum)
            z += 1
            for d in shuffle_idx:
                doc = docs[d]
                user = users[d]
                item = items[d]
                rating = ratings[d]
                doclen = len(doc)
                for i, w in enumerate(doc):
                    if i + span >= doclen:
                        continue

                    if i < span:
                        continue
                    #
                    #                    print(i,span,sample_idx,context_window)
                    #                    print(span,context_window,sample_idx,i,batch_size,
                    #                          batch_word_data[sample_idx*context_window : sample_idx*context_window+context_window],
                    #                          doc[i-span:i]+ doc[i+1:i+1+span],
                    #                          '-'*20)
                    batch_word_data[sample_idx * context_window: sample_idx * context_window + context_window] \
                        = doc[i - span:i] + doc[i + 1:i + 1 + span]

                    # batch_word_data = np.reshape(batch_word_data,(-1,1))
                    batch_labels[sample_idx] = [w]
                    batch_doc[sample_idx] = d
                    batch_users[sample_idx] = user
                    batch_items[sample_idx] = item
                    batch_ratings[sample_idx] = rating
                    batch_inverse_weight[sample_idx] = len(doc)- context_window
                    user_bias_with_mean_batch[sample_idx] = user_bias_with_mean[user]
                    item_bias_with_mean_and_user_batch[sample_idx] = item_bias_with_mean_and_user[item]

                    if sample_idx == batch_size - 1:
                        sample_idx = 0

                        yield batch_labels, \
                              batch_word_data, \
                              batch_doc, \
                              batch_users, \
                              batch_items, \
                              batch_ratings, \
                              batch_inverse_weight,\
                              user_bias_with_mean_batch, \
                              item_bias_with_mean_and_user_batch



                    else:
                        sample_idx += 1
                        # print('z=====================',z)


data = ReviewRatingData('Amazon_Instant_Video_5.json')
# data=ReviewRatingData('test.json')


# =============================================================================
# 矩阵分解，用神经网络来实现
# =============================================================================

from tensorlayer.utils import dict_to_one
from tensorlayer.layers import list_remove_repeat


def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high, dtype=tf.float32)
#
# class RandomUniform(tf.Initializer):
#   """Initializer that generates tensors with a uniform distribution.
#
#   Args:
#     minval: A python scalar or a scalar tensor. Lower bound of the range
#       of random values to generate.
#     maxval: A python scalar or a scalar tensor. Upper bound of the range
#       of random values to generate.  Defaults to 1 for float types.
#     seed: A Python integer. Used to create random seeds. See
#       @{tf.set_random_seed}
#       for behavior.
#     dtype: The data type.
#   """
#
#   def __init__(self, minval=0, maxval=None, seed=None, dtype=dtypes.float32):
#     self.minval = minval
#     self.maxval = maxval
#     self.seed = seed
#     self.dtype = dtypes.as_dtype(dtype)
#
#   def __call__(self, shape, dtype=None, partition_info=None):
#     if dtype is None:
#       dtype = self.dtype
#     return random_ops.random_uniform(shape, self.minval, self.maxval,
#                                      dtype, seed=self.seed)
#
#   def get_config(self):
#     return {"minval": self.minval,
#             "maxval": self.maxval,
#             "seed": self.seed,
#             "dtype": self.dtype.name}
#



class AddLayer(tl.layers.Layer):
    def __init__(
            self,
            name='addition_layer',
            *layers
    ):
        '''
        输入若干个outputs.shape一样的列向量（或为标量），输出为element-wise 相加，另加一个常数。参数只有1个。
        '''
        # check layer name (fixed)

        tl.layers.Layer.__init__(self, name=name)
        print("  [TL] AddLayer %s:" % (self.name))
        outputs = [x.outputs for x in layers]
        self.inputs = tf.concat(outputs, 0)

        # the input of this layer is the output of previous layer (fixed)



        # operation (customized)
        # 在末端加个1，代表bias项。所以b向量的最后一列就是商品bias。a向量的第一项类似

        with tf.variable_scope(name) as vs:
            c = tf.get_variable(name='total_mean_bias', shape=[1], initializer=tf.constant_initializer(value=0.0))

            s = tf.reduce_sum(outputs, 0)
            self.outputs = tf.reshape(s + c, [-1, 1])




        #        self.outputs = tf.reshape(tf.reduce_sum(layer1.outputs*layer2.outputs,1) ,[-1,1])

        # get stuff from previous layer (fixed)

        self.all_layers = []
        self.all_params = []
        self.all_drop = {}
        for x in layers:
            self.all_layers.extend(list(x.all_layers))
            self.all_params.extend(list(x.all_params))
            self.all_drop.update(dict(x.all_drop))

        self.all_params.append(c)
        # update layer (customized)
        self.all_layers.extend([self.outputs])


class MultiLayer(tl.layers.Layer):
    '''
    注意：在梯度下降时对该层需要用交替梯度下降。
    '''

    def __init__(
            self,
            layer1,
            layer2,
            name='multi_layer'
    ):
        # check layer name (fixed)
        tl.layers.Layer.__init__(self, name=name)
        print("  [TL] MultiLayer %s:" % (self.name))

        # the input of this layer is the output of previous layer (fixed)
        self.inputs = tf.concat([layer1.outputs, layer2.outputs], 0)

        self.outputs = tf.reshape(tf.reduce_sum(layer1.outputs * layer2.outputs, 1), [-1, 1])

        # get stuff from previous layer (fixed)
        self.all_layers = list(layer1.all_layers)
        self.all_layers.extend(list(layer2.all_layers))

        self.all_params = list(layer1.all_params)
        self.all_params.extend(list(layer2.all_params))

        self.all_params1 = list(layer1.all_params)
        self.all_params2 = list(layer2.all_params)

        self.all_drop = dict(layer1.all_drop)
        self.all_drop.update(dict(layer2.all_drop))

        # update layer (customized)
        self.all_layers.extend([self.outputs])


class EmbeddingInputlayer_DynamicShape(tl.layers.Layer):
    def __init__(
            self,
            inputs,
            vocabulary_size,  # could be a placeholder,scalar
            embedding_size,  # also a ph
            # E_init = tf.random_uniform_initializer(-0.1, 0.1),
            # E_init_args = {},
            name='embedding_layer',
    ):
        tl.layers.Layer.__init__(self, name=name)
        self.inputs = inputs
        print("  [TL] EmbeddingInputlayer %s:" % (self.name))

        with tf.variable_scope(name) as vs:
            init = tf.random_uniform((vocabulary_size, embedding_size), -0.1, 0.1)
            embeddings = tf.get_variable('embeddings', initializer=init, validate_shape=False)
            embeddings.set_shape(embeddings.get_shape())

            #            embeddings = tf.get_variable(name='embeddings',
            #                                    shape=(vocabulary_size, embedding_size),
            #                                    initializer=E_init,
            #                                    **E_init_args)
            embed = tf.nn.embedding_lookup(embeddings, self.inputs)

        self.outputs = embed

        self.all_layers = [self.outputs]
        self.all_params = [embeddings]
        self.all_drop = {}


class NCElosslayer(tl.layers.Layer):

    def __init__(
        self,
        layer = None,
        n_classes=100,
        labels = None,
        num_sampled = 64,
        W_init = tf.truncated_normal_initializer(stddev=0.1),
        b_init = tf.constant_initializer(value=0.0),
        W_init_args = {},
        b_init_args = {},
        name ='nceloss_layer',
    ):
        tl.layers.Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        if self.inputs.get_shape().ndims != 2:
            raise Exception("The input dimension must be rank 2, please reshape or flatten it")

        n_in = int(self.inputs.get_shape()[-1])
        self.n_units = n_classes
        print("  [TL] NCELossLayer  %s: %d " % (self.name, self.n_units))
        with tf.variable_scope(name) as vs:
            W = tf.get_variable(name='W', shape=(n_in, n_classes), initializer=W_init, **W_init_args)
            if b_init is not None:
                try:
                    b = tf.get_variable(name='b', shape=(n_classes), initializer=b_init, **b_init_args)
                except: # If initializer is a constant, do not specify shape.
                    b = tf.get_variable(name='b', initializer=b_init, **b_init_args )

            else:
                assert(0)

            self.outputs = tf.nn.nce_loss(weights=W,
                                          biases=b,
                                          labels=labels,
                                          inputs=layer.outputs,
                                          num_sampled=num_sampled,
                                          num_classes=n_classes)

        # Hint : list(), dict() is pass by value (shallow), without them, it is
        # pass by reference.
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )
        if b_init is not None:
            self.all_params.extend( [W, b] )
        else:
            self.all_params.extend( [W] )




class AbsModel:
    def __init__(self,data):
        self.data=data
        # self.uv_embedding_size = uv_embedding_size # Dimension of user/item embedding vector.
        #self.uv_latentfactor_size=uv_latentfactor_size
        # self.num_sampled = num_sampled
        # self.word_embedding_size=word_embedding_size  # Dimension of word/doc embedding vector.
        #self.rating_softmax_width = uv_latentfactor_size
        # self.word_softmax_width = word_embedding_size
        # self.context_window = context_window
        self.vocabulary_size=data.getVocabularyNum()
        self.num_users=data.getUserNum()
        self.num_items=data.getItemNum()
        self.num_ratings_val=data.getCVNum()




        return


class MFModel(AbsModel):
    def __init__(self,data):

        super(MFModel, self).__init__(data)


        return

    def buildNetWork(self, uv_latentfactor_size=256, uv_embedding_size=128, word_embedding_size=128, activateFunction=None):

        self.uv_latentfactor_size = uv_latentfactor_size
        self.uv_embedding_size = uv_embedding_size
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.name_scope('MFModel'):
                # batch_size = tf.placeholder(tf.int32)

                self.inputUserid = tf.placeholder(tf.int32)
                self.inputItemid = tf.placeholder(tf.int32)
                self.inputContextWordid = tf.placeholder(tf.int32)  # test的时候用不到
                self.label_wordid = tf.placeholder(tf.int32)  # 同上
                self.label_rating = tf.placeholder(tf.float32)
                self.inv_weight = tf.placeholder(tf.float32)
                self.para_lambda = tf.placeholder(tf.float32)
                self.para_learning_rate = tf.placeholder(tf.float32)
                # self.para_uv_latentfactor_size = tf.placeholder(tf.int32)



                # =============================================================================
                #                  #长度+1，代表bia项。ULF的第一项就是用户bias
                #                 ULF = tl.layers.EmbeddingInputlayer(self.inputUserid,
                #                                                      self.num_users,
                #                                                      self.uv_latentfactor_size+1,
                #                                                      name='userLatentFactorLayer',
                #                                                      E_init = tf.truncated_normal_initializer(stddev=0.2)
                #                                                      )
                #                 VLF = tl.layers.EmbeddingInputlayer(self.inputItemid,
                #                                                      self.num_items,
                #                                                      self.uv_latentfactor_size+1,
                #                                                      name='itemLatentFactorLayer',
                #                                                      E_init = tf.truncated_normal_initializer(stddev=0.2)
                #                                                      )
                # =============================================================================
                ULF = tl.layers.EmbeddingInputlayer(self.inputUserid,
                                                    self.num_users,
                                                    self.uv_latentfactor_size,
                                                    name='userLatentFactorLayer',
                                                    # E_init=tf.truncated_normal_initializer(stddev=0.2)
                                                    )
                # =============================================================================
                #                 ULF = EmbeddingInputlayer_DynamicShape(self.inputUserid,
                #                                                     self.num_users,
                #                                                     self.para_uv_latentfactor_size,
                #                                                     name='userLatentFactorLayer',
                #                                                     #E_init = tf.truncated_normal_initializer(stddev=0.2)
                #                                                     )
                # =============================================================================

                Ubias = tl.layers.EmbeddingInputlayer(self.inputUserid,
                                                      self.num_users,
                                                      1,
                                                      name='userBias',
                                                      E_init=tf.truncated_normal_initializer(stddev=0)
                                                      )

                VLF = tl.layers.EmbeddingInputlayer(self.inputItemid,
                                                    self.num_items,
                                                    self.uv_latentfactor_size,
                                                    name='itemLatentFactorLayer',
                                                    # E_init=tf.truncated_normal_initializer(stddev=0.2)
                                                    )
                # =============================================================================
                #                 VLF = EmbeddingInputlayer_DynamicShape(self.inputItemid,
                #                                                     self.num_items,
                #                                                     self.para_uv_latentfactor_size,
                #                                                     name='itemLatentFactorLayer',
                #                                                     #E_init = tf.truncated_normal_initializer(stddev=0.2)
                #                                                     )
                #
                # =============================================================================
                Vbias = tl.layers.EmbeddingInputlayer(self.inputItemid,
                                                      self.num_items,
                                                      1,
                                                      name='itemBias',
                                                      E_init=tf.truncated_normal_initializer(stddev=0)
                                                      )

                # self.user_latent_factor=ULF.all_params[0]
                # self.item_latent_factor=VLF.all_params[0]
                with tf.variable_scope('userLatentFactorLayer', reuse=True):
                    self.user_lf_mat = tf.get_variable(name='embeddings')
                with tf.variable_scope('itemLatentFactorLayer', reuse=True):
                    self.item_lf_mat = tf.get_variable(name='embeddings')

                with tf.variable_scope('userBias', reuse=True):
                    self.user_bias = tf.get_variable(name='embeddings')
                with tf.variable_scope('itemBias', reuse=True):
                    self.item_bias = tf.get_variable(name='embeddings')


                    # print(self.user_lf_mat)

                self.network = MultiLayer(ULF, VLF, name='multiplyLayer')  # 无参数

                # 用于下面的交替梯度下降

                #                first_vars = list(self.network.all_params1)
                #                unassigned_first_vars = list(self.network.all_params1)
                #                second_vars = list(self.network.all_params2)
                #                unassigned_second_vars = list(self.network.all_params2)

                #                if self.uv_latentfactor_size == 0:
                #                    self.network = AddLayer('additionLayer',Ubias,Vbias)
                #                else:

                self.network = AddLayer('additionLayer', self.network, Ubias, Vbias)

                with tf.variable_scope('additionLayer', reuse=True):
                    self.total_bias = tf.get_variable(name='total_mean_bias')




                #                first_vars.append(self.user_bias)
                #                first_vars.append(self.item_bias)
                #                first_vars.append(self.total_bias)
                #                second_vars.append(self.user_bias)
                #                second_vars.append(self.item_bias)
                #                second_vars.append(self.total_bias)
                #                totalbias_vars = [self.total_bias]
                #                assignTBtrainUVB_vars = [self.user_bias,self.item_bias]
                #

                self.predictRating = self.network.outputs
                # To print all attributes of a Layer.
                # attrs = vars(network)
                # print(', '.join("%s: %s\n" % item for item in attrs.items()))
                # print(network.all_drop)

                def weighted_mean_squared_error(output,target,weight):
                    assert(output.get_shape().ndims == 2)
                    e = tf.reduce_sum(tf.squared_difference(output, target), 1)
                    mse = tf.reduce_sum(weight * e) / tf.reduce_sum(weight)
                    return mse

                self.cost = weighted_mean_squared_error(self.predictRating, self.label_rating,tf.reciprocal(self.inv_weight,name='reciprocal'))



                # 正则项：
                self.l2 = tf.contrib.layers.l2_regularizer(self.para_lambda)(self.user_lf_mat)
                self.l2 += tf.contrib.layers.l2_regularizer(self.para_lambda)(self.item_lf_mat)

                self.cost = self.cost + self.l2
                good = tf.less(tf.abs(self.predictRating - self.label_rating), 0.5)
                self.accuracy = tf.reduce_mean(tf.cast(good, tf.float32))




            #                print(first_vars,second_vars)
            #                print('tf.trainable_vars:',tf.trainable_variables())
            #                print('tf.all_vars:',tf.all_variables())


            #                self.first_vars = first_vars
            #                self.second_vars = second_vars
            #                self.unassigned_first_vars = unassigned_first_vars
            #                self.unassigned_second_vars = unassigned_second_vars
            #
            #                self.user_lf_mat.set_shape((self.num_users,uv_latentfactor_size))
            #                self.item_lf_mat.set_shape((self.num_items,uv_latentfactor_size))



            #                self.train_op1 = tf.train.AdamOptimizer(self.para_learning_rate).minimize(self.cost,var_list = first_vars,global_step=self.global_step)
            #                self.train_op2 = tf.train.AdamOptimizer(self.para_learning_rate).minimize(self.cost,var_list = second_vars,global_step=self.global_step)
            #
            #
            #                print(unassigned_first_vars)
            #                print(unassigned_second_vars)
            #                self.train_op_ass1 = tf.train.GradientDescentOptimizer(self.para_learning_rate).minimize(self.cost,var_list = unassigned_first_vars,global_step=self.global_step)
            #                self.train_op_ass2 = tf.train.GradientDescentOptimizer(self.para_learning_rate).minimize(self.cost,var_list = unassigned_second_vars,global_step=self.global_step)
            #                self.train_tb_op = tf.train.GradientDescentOptimizer(self.para_learning_rate).minimize(self.cost,var_list = totalbias_vars,global_step=self.global_step)
            #
            #                self.train_op_assignTB_trainUVB = tf.train.GradientDescentOptimizer(self.para_learning_rate).minimize(self.cost,var_list = assignTBtrainUVB_vars,global_step=self.global_step)
            #

    def fit(self,  batch_size=float('+inf'), context_window=6, n_epoch=200, para_lambda=1.5, #num_sampled=64,
            optimizer=tf.train.AdamOptimizer, para_learning_rate=0.001, lr_decay=False, valandsave_epoch_freq=5,
            tensorboard=False, tensorboard_epoch_freq=5, tensorboard_weight_histograms=True, tensorboard_graph_vis=True,
            TrainMode=0, ContinueTraining=False, plotResults=True, AutoConvergence=True
            ):
        '''

        :param batch_size:
        :param context_window:
        :param n_epoch:
        :param para_lambda:
        :param optimizer:
        :param para_learning_rate:
        :param lr_decay:
        :param valandsave_epoch_freq:
        :param tensorboard:
        :param tensorboard_epoch_freq:
        :param tensorboard_weight_histograms:
        :param tensorboard_graph_vis:
        :param TrainMode:
            0:  Normal
            1:  Train Bias Only
            2:  Not Train user/item/total Bias
        :param ContinueTraining:
        :param plotResults:
        :param AutoConvergence:
        :return:
        '''

        #self.data = data
        #self.uv_embedding_size = uv_embedding_size # Dimension of user/item embedding vector.
        #self.uv_latentfactor_size=uv_latentfactor_size
        #self.num_sampled = num_sampled
        #self.word_embedding_size=word_embedding_size  # Dimension of word/doc embedding vector.
        #self.rating_softmax_width = uv_latentfactor_size
        #self.word_softmax_width = word_embedding_size
        #self.context_window = context_window
        self.vocabulary_size=data.getVocabularyNum()
        self.num_users=data.getUserNum()
        self.num_items=data.getItemNum()
        self.num_ratings_val=data.getCVNum()
        for num_training_sample,_ in enumerate(data.generate_batch_NOreviewdata(1)):
            pass

        self.num_training_sample = num_training_sample
        print('训练样本总数为',num_training_sample)
        #assert(batch_size < num_training_sample,'batchsize不能超过样本总数！')
        if batch_size > num_training_sample:
            batch_size = num_training_sample

        step_per_ep= num_training_sample // batch_size
        # step_per_ep = (self.data.getTrainNum()//batch_size)+1 
        def func(x, a, b, c, d):  # 指数衰减
            return a * np.exp(-b * (x + d)) + c

        def func_dev(x, a, b, c, d):  # 上面那个函数的导数
            return a * np.exp(-b * (x + d)) * (-b) * x

        def polyfit(x, y):
            results = {}

            try:
                popt, pcov = curve_fit(func, x, y, bounds=(0, [np.inf, np.inf, np.inf, np.inf]))

                results['fit_params'] = popt

                # r-squared
                yhat = func(x, *popt)  # 预测值
                ybar = np.sum(y) / len(y)  # y的均值
                ssreg = np.sum((yhat - ybar) ** 2)  # ybar = mean(yhat) sum([ (yihat - ybar)**2 for yihat in yhat])
                sstot = np.sum((y - ybar) ** 2)  # or sum([ (yi - ybar)**2 for yi in y])
                results['determination'] = ssreg / sstot
                return results

            except:

                return {'fit_params': [0, 0, 0, 0], 'determination': 0}

        with tf.Session(graph=self.graph) as sess:

            if (tensorboard):#tensorboard的初始化
                print("Setting up tensorboard ...")
                # Set up tensorboard summaries and saver
                tl.files.exists_or_mkdir('logs/')

                # Only write summaries for more recent TensorFlow versions
                if hasattr(tf, 'summary') and hasattr(tf.summary, 'FileWriter'):
                    if tensorboard_graph_vis:
                        train_writer = tf.summary.FileWriter('logs/train', sess.graph)
                        val_writer = tf.summary.FileWriter('logs/validation', sess.graph)
                    else:
                        train_writer = tf.summary.FileWriter('logs/train')
                        val_writer = tf.summary.FileWriter('logs/validation')

                # Set up summary nodes
                if (tensorboard_weight_histograms):
                    for param in self.network.all_params:
                        if hasattr(tf, 'summary') and hasattr(tf.summary, 'histogram'):
                            print('Param name ', param.name)
                            tf.summary.histogram(param.name, param)

                if hasattr(tf, 'summary') and hasattr(tf.summary, 'histogram'):
                    tf.summary.scalar('cost', self.cost)
                    tf.summary.scalar('acc', self.accuracy)
                    # tf.summary.scalar('costval',costval)

                merged = tf.summary.merge_all()

                # Initalize all variables and summaries

                print("Finished! use $tensorboard --logdir=logs/ to start server")

            # 初始化,构建训练变量列表
            with self.graph.as_default():
                with tf.name_scope('MFModel'):

                    if TrainMode == 1:  # 只训练线性偏差
                        first_vars = [self.user_bias, self.item_bias, self.total_bias]
                        second_vars = first_vars
                        all_vars = first_vars

                    elif TrainMode == 2:  # 用之前训练好的线性偏差
                        first_vars = [self.user_lf_mat]
                        second_vars = [self.item_lf_mat]
                        all_vars = [self.user_lf_mat, self.item_lf_mat]
                    else:
                        first_vars = [self.user_lf_mat, self.user_bias, self.item_bias, self.total_bias]
                        second_vars = [self.item_lf_mat, self.user_bias, self.item_bias, self.total_bias]
                        all_vars = [self.user_lf_mat, self.item_lf_mat, self.user_bias, self.item_bias, self.total_bias]


                    self.global_step = tf.Variable(0, trainable=False)

                    # =============================================================================
                    #                     def reset_shape(v):
                    #                         from tensorflow.python.framework import tensor_shape
                    #                         v._ref().set_shape(tensor_shape.unknown_shape())
                    #                         v.value().set_shape(tensor_shape.unknown_shape())
                    #                         #tensor._shape = tensor_shape.unknown_shape()
                    #
                    #                     reset_shape(self.user_lf_mat)
                    #                     reset_shape(self.item_lf_mat)
                    # =============================================================================

                    # self.user_lf_mat.set_shape([self.num_users, self.uv_latentfactor_size])
                    # self.item_lf_mat.set_shape([self.num_items, self.uv_latentfactor_size])

                    # 计算梯度
                    if lr_decay == True:
                        self.learning_rate = tf.train.exponential_decay(para_learning_rate,
                                                                        global_step = self.global_step,
                                                                        decay_steps = step_per_ep * n_epoch,
                                                                        decay_rate= 0.96,
                                                                        # staircase=True
                                                                        )
                        optimizer_op = optimizer(self.learning_rate)


                    else:
                        optimizer_op = optimizer(para_learning_rate)


                    grads_and_vars = optimizer_op.compute_gradients(self.cost, var_list=all_vars)
                    grads_and_vars1 = [(g, v) for g, v in grads_and_vars if v in first_vars]
                    grads_and_vars2 = [(g, v) for g, v in grads_and_vars if v in second_vars]

                       # grads_and_vars1 = optimizer_op.compute_gradients( self.cost,var_list = first_vars)
                       # grads_and_vars2 = optimizer_op.compute_gradients( self.cost,var_list = second_vars)
                       #
                       # vars_with_grad1 = [v for g, v in grads_and_vars1 if g is not None]
                       # vars_with_grad2 = [v for g, v in grads_and_vars2 if g is not None]
                       # if (not vars_with_grad1) or (not vars_with_grad2):
                       #     raise ValueError("No gradients provided for any variable")


                    # grad_ops = [tf.reduce_mean(tf.convert_to_tensor(g)) for g, v in grads_and_vars]
                    grad_ops = [tf.reduce_max(tf.convert_to_tensor(g)) for g, v in grads_and_vars]
                    train_op1 = optimizer_op.apply_gradients(grads_and_vars1, global_step=self.global_step)
                    train_op2 = optimizer_op.apply_gradients(grads_and_vars2, global_step=self.global_step)


                    #
                    #
                    # train_op1 = optimizer(self.para_learning_rate).minimize(self.cost,var_list = first_vars,global_step=self.global_step)
                    # train_op2 = optimizer(self.para_learning_rate).minimize(self.cost,var_list = second_vars,global_step=self.global_step)

                    # grad_op = tf.gradients(self.cost,all_vars)

            tl.layers.initialize_global_variables(sess)
            # sess.run(tf.global_variables_initializer(),
            #          feed_dict={self.para_uv_latentfactor_size: uv_latentfactor_size})

            if TrainMode == 1:
                self._setzero_uvmat(sess)
            if TrainMode == 2:
                self._assign_uvtbias(sess)

            if ContinueTraining:
                self._assign_allVars(sess)

            print("Start training the network ...")
            start_time_begin = time.time()
            tensorboard_train_index, tensorboard_val_index = 0, 0
            # batchs = self.data.generate_batch(batch_size,context_window)
            trainlosses = []
            trainlosses_fit=[]
            vallosses = []
            convergence_flag= False
            self.exitcode = -2
            # break_flag = False
            # curvefit_flag = False
            # pltskip = float('+inf')

            for epoch in range(n_epoch):
                start_time = time.time()
                loss_ep = 0
                l2_ep = 0
                n_step = 0
                grad_ep = np.array([0. for i in range(len(all_vars))])

                #                for i in range(step_per_ep):
                #                    (batch_labels, batch_word_data, batch_doc, batch_users, batch_items, batch_ratings,
                #                _, _)   = next(batchs)
                for i, (batch_users, batch_items, batch_ratings,
                        _, _) in enumerate(self.data.generate_batch_NOreviewdata(batch_size)):

                    # print('step %d of %d,lr:%f' % (int(self.global_step.eval(session=sess)),
                    #                                int(step_per_ep * n_epoch),
                    #                                float(self.learning_rate.eval(session=sess))
                    #                                ))
                    a1,a2,a3,a4,a5,a6 = sess.run([tf.contrib.layers.l2_regularizer(1.0)(self.user_lf_mat),
                              tf.contrib.layers.l2_regularizer(1.0)(self.item_lf_mat),
                              tf.reduce_sum(self.user_lf_mat*self.user_lf_mat),
                              tf.reduce_sum(self.item_lf_mat * self.item_lf_mat),
                              tf.shape(self.item_lf_mat),
                              tf.reduce_mean(self.item_lf_mat)
                              ])
                    print(a1,a2,a3,a4,a5,a6)


                    self.l2 = tf.contrib.layers.l2_regularizer(self.para_lambda)(self.user_lf_mat)
                    self.l2 += tf.contrib.layers.l2_regularizer(self.para_lambda)(self.item_lf_mat)

                    feed_dict = {self.inputUserid: batch_users,
                                 self.inputItemid: batch_items,
                                 # self.inputContextWordid : batch_word_data,
                                 # self.label_wordid : batch_ratings,
                                 self.label_rating: batch_ratings,
                                 self.para_lambda: para_lambda,
                                 #self.para_learning_rate: lr,
                                 #self.inv_weight: batch_inv_weight
                                 self.inv_weight: [1.0 for _ in range(batch_size)]
                                 }

                    feed_dict.update(self.network.all_drop)  # enable noise layers
                    if i % 2 == 0:
                        l2, loss, _ ,grad = sess.run([self.l2, self.cost, train_op1, grad_ops], feed_dict=feed_dict)
                        # loss, _ = sess.run([self.cost, train_op1], feed_dict=feed_dict)

                    elif i % 2 == 1:
                        l2, loss, _ ,grad = sess.run([self.l2, self.cost, train_op2, grad_ops], feed_dict=feed_dict)
                        # loss, _ = sess.run([self.cost, train_op2], feed_dict=feed_dict)

                    loss_ep += loss
                    l2_ep += l2
                    grad_ep += np.array([np.mean(g) for g in grad])

                    n_step += 1




                # =============================================================================
                # g.values有bug吗？
                #                     for g in grad:
                #                          #print(g)
                #                          print('====================================')
                #                          print(g.values)
                #                          print(g.values.shape)
                #                          print(g.indices)
                #                          print(g.indices.shape)
                #                          print(g.dense_shape)
                # grads = [tf.convert_to_tensor(grad_ops1[i]) for i in ]
                #                    for i in range(4) :
                #                        g =sess.run(tf.convert_to_tensor(grad_ops1[i]),feed_dict=feed_dict)
                #                        print(g)
                #                        print(g.shape)
                # =============================================================================

                # 每个epoch执行一次：
                # pltskip = 15
                # if (epoch == pltskip):
                #     trainlosses = []
                loss_ep = loss_ep / n_step
                l2_ep = l2_ep / n_step
                # if loss_ep < 30:
                #     curvefit_flag = True
                #     pltskip = epoch

                trainlosses.append(float(loss_ep))
                grad_ep = grad_ep/ n_step
                if (epoch) % valandsave_epoch_freq == 0:
                    print("Epoch %d of %d took %fs, loss %f, loss2 %f, maxgrad %f" % (epoch + 1, n_epoch, time.time() - start_time, loss_ep,loss_ep-l2_ep, max(grad_ep)))




                # 每valandsave_epoch_freq 个epoch执行一次
                if (epoch ) % valandsave_epoch_freq == 0:

                    # 验证集上的cost
                    dp_dict = dict_to_one(self.network.all_drop)  # disable noise layers
                    feed_dict = {self.inputUserid: data.getCVData('user'),
                                 self.inputItemid: data.getCVData('item'),
                                 # self.inputContextWordid : batch_word_data,
                                 # self.label_wordid : batch_ratings,
                                 self.label_rating: np.reshape(np.array(data.getCVData('rating')), (-1, 1)),
                                 self.inv_weight: [1.0 for _ in range(data.getCVNum())],
                                 self.para_lambda: 0.0
                                 # self.para_uv_latentfactor_size : uv_latentfactor_size
                                 }

                    feed_dict.update(dp_dict)
                    # grad_val, loss_val = sess.run([grad_ops, self.cost], feed_dict=feed_dict)  # cost\acc
                    loss_val = sess.run(self.cost, feed_dict=feed_dict)
                    print('validation loss:', loss_val)
                    vallosses.append(float(loss_val))

                    # 存档
                    self.final_user_lf_mat = self.user_lf_mat.eval()
                    self.final_item_lf_mat = self.item_lf_mat.eval()
                    self.final_total_bias = self.total_bias.eval()
                    self.final_user_bias = self.user_bias.eval()
                    self.final_item_bias = self.item_bias.eval()
                    self.final_train_cost = loss_ep
                    self.final_val_cost = loss_val

                # 判定是否收敛,同样每个EP执行一次
                # print(loss_ep)
                if(loss_ep==float('inf') or math.isnan(loss_ep)):
                    self.exitcode = 0
                    break

                if ( loss_ep < 3 )  or convergence_flag == True or time.time() - start_time_begin > 3600:

                    # if (max(grad_ep) < 1e-4):
                    #     print('从梯度上看好像收敛了！当前步数为：', epoch + 1)
                    #     break_flag = True
                    #     break

                    # try:
                    if time.time() - start_time_begin > 5400: #最长训练时间
                        self.exitcode = -1
                        break
                    convergence_flag = True
                    trainlosses_fit.append(loss_ep)
                    # print(len(trainlosses_fit))
                    if len(trainlosses_fit) > 10 :# 足够长才能拟合
                        re = polyfit(np.arange(len(trainlosses_fit)), np.array(trainlosses_fit))
                        ydev = func_dev(len(trainlosses_fit), *re['fit_params'])
                        #print('拟合优度：', re['determination'])
                        print('拟合斜率：', ydev)
                        if re['determination'] > 0.99 and abs(ydev) < 1e-4:  # 好像收敛了
                            print('从拟合曲线上看好像收敛了！当前步数为：', epoch + 1)
                            # print(re['fit_params'])
                            # break_flag = True
                            if AutoConvergence:
                                self.exitcode = 0
                                break
                    # =============================================================================
                    #                 #计算梯度：在某个批上计算
                    #                 grad = sess.run(grad_op, feed_dict=feed_dict)
                    #                 tf.IndexedSlices
                    #                 #print(grad)
                    #                 g_ep=0
                    #                 for g in grad:
                    #                     #print(g)
                    #                     print('====================================')
                    #                     print(g.values)
                    #                     print(g.values.shape)
                    #                     print(g.indices)
                    #                     print(g.indices.shape)
                    #                     print(g.dense_shape)
                    #
                    #                     gm = np.mean(g.values)
                    #                     print('average grad:',gm)
                    #                     #g_ep += g
                    #
                    #                 #print('all average grad:',g_ep/)
                    # =============================================================================

                # 每tensorboard_epoch_freq个epoch，记录tensorboard的信息。训练集上的cost等。
                if tensorboard and hasattr(tf, 'summary'):
                    if epoch + 1 == 1 or (epoch + 1) % tensorboard_epoch_freq == 0:
                        # for i in range(step_per_ep):
                        #     (batch_labels, batch_word_data, batch_doc, batch_users, batch_items, batch_ratings,
                        #      _, _) = next(batchs)
                       for (batch_users, batch_items, batch_ratings,
                       _, _) in data.generate_batch_NOreviewdata(batch_size):

                            dp_dict = dict_to_one(self.network.all_drop)  # disable noise layers
                            feed_dict = {self.inputUserid: batch_users,
                                         self.inputItemid: batch_items,
                                         # self.inputContextWordid : batch_word_data,
                                         # self.label_wordid : batch_ratings,
                                         self.label_rating: batch_ratings,
                                         self.para_lambda: 0.0
                                         # self.para_uv_latentfactor_size : uv_latentfactor_size
                                         }

                            feed_dict.update(dp_dict)
                            result = sess.run(merged, feed_dict=feed_dict)
                            train_writer.add_summary(result, tensorboard_train_index)
                            tensorboard_train_index += 1

#                        if (True):  # 同时在val集上验证
                            dp_dict = dict_to_one(self.network.all_drop)  # disable noise layers

                            feed_dict = {self.inputUserid: data.getCVData('user'),
                                         self.inputItemid: data.getCVData('item'),
                                         # self.inputContextWordid : batch_word_data,
                                         # self.label_wordid : batch_ratings,
                                         self.label_rating: np.reshape(np.array(data.getCVData('rating')), (-1, 1)),
                                         self.inv_weight: [1.0 for _ in range(data.getCVNum())],
                                         self.para_lambda: 0.0
                                         # self.para_uv_latentfactor_size : uv_latentfactor_size
                                         }

                            feed_dict.update(dp_dict)
                            result = sess.run(merged, feed_dict=feed_dict)  # cost\acc
                            val_writer.add_summary(result, tensorboard_val_index)
                            tensorboard_val_index += 1



                            # self.test(sess=sess)


            #结束循环
            self.final_val_cost_min = min(vallosses)
            self.training_vallosses = vallosses
            self.training_trainlosses = trainlosses

            if plotResults:

                plt.plot(trainlosses, 'b.-', label='train_loss')
                valx=[x*valandsave_epoch_freq  for x in range(len(vallosses))]
                plt.plot(valx,vallosses, 'r.-', label='val_loss')

                # re = polyfit(np.arange(len(trainlosses)), np.array(trainlosses))
                # print('拟合优度：', re['determination'])
                # trainlosses_hat = func(np.arange(len(trainlosses)), *re['fit_params'])
                # trainlosses_hat = map(lambda x:func(x,*re['fit_params']),np.arange(len(trainlosses)))
                # plt.plot(trainlosses_hat, 'g--', label='train_loss_fit')
                plt.xlabel('eproch')
                plt.ylabel('loss')
                plt.legend()
                plt.title('n_epoch:%d,lf:%d,bs:%d,lambda:%f,lr:%f' % (n_epoch,self.uv_latentfactor_size, batch_size, para_lambda, para_learning_rate) )
                plt.show()

            #TODO:subplot

            print("Total training time: %fs" % (time.time() - start_time_begin))


    def saveModel(self, path='mfmodel.npy'):
        tl.files.save_any_to_npy(save_dict={'ulf': self.final_user_lf_mat,
                                            'vlf': self.final_item_lf_mat,
                                            'ub': self.final_user_bias,
                                            'vb': self.final_item_bias,
                                            'tb': self.final_total_bias}, name=path)

    def loadModel(self, path='mfmodel.npy'):
        dic = tl.files.load_npy_to_any(name=path)

        self.final_user_lf_mat = dic['ulf']
        self.final_item_lf_mat = dic['vlf']
        self.final_user_bias = dic['ub']
        self.final_item_bias = dic['vb']
        self.final_total_bias = dic['tb']

    def _assign_allVars(self, sess):

        assign_op = [self.user_lf_mat.assign(self.final_user_lf_mat),
                     self.item_lf_mat.assign(self.final_item_lf_mat),
                     self.total_bias.assign(self.final_total_bias),
                     self.user_bias.assign(self.final_user_bias),
                     self.item_bias.assign(self.final_item_bias)
                     ]
        sess.run(assign_op)

    def _setzero_uvmat(self, sess):
        assign_op = [self.user_lf_mat.assign(tf.zeros(tf.shape(self.user_lf_mat))),
                     self.item_lf_mat.assign(tf.zeros(tf.shape(self.item_lf_mat)))
                     ]
        sess.run(assign_op)

    def _assign_uvtbias(self, sess):
        assign_op = [
                     self.total_bias.assign(self.final_total_bias),
                     self.user_bias.assign(self.final_user_bias),
                     self.item_bias.assign(self.final_item_bias)
                     ]
        sess.run(assign_op)

    def _assign_TB(self, sess):
        _, _, mean_rating, _ = self.data.getTrainMeanRating()
        t = np.array(mean_rating)
        t = t.reshape([1])

        assign_op = [self.total_bias.assign(t)
                     ]
        sess.run(assign_op)

    def _assign_linear_var(self, sess):

        user_bias, item_bias, mean_rating, item_mean_term2 = self.data.getTrainMeanRating()
        user_bias = [i - mean_rating for i in user_bias]
        # item_bias = [i - mean_rating for i in item_bias]
   
        item_bias = [i - j for (i, j) in zip(item_bias, item_mean_term2)]

        u = np.array(user_bias)
        u = u.reshape([-1, 1])

        v = np.array(item_bias)
        v = v.reshape([-1, 1])

        t = np.array(mean_rating)
        t = t.reshape([1])

        assign_op = [self.user_bias.assign(u),
                     self.item_bias.assign(v),
                     self.total_bias.assign(t)
                     ]

        sess.run(assign_op)

    def test(self, bLinear=False, TestinTrainset=False, sess=None):

        # print('Start testing the network ...')


        data = self.data
        cost = self.cost
        acc = self.accuracy

        if sess == None:
            with tf.Session(graph=self.graph) as sess:
                tl.layers.initialize_global_variables(sess)
                # sess.run(tf.global_variables_initializer(),
                #          feed_dict={self.para_uv_latentfactor_size: self.uv_latentfactor_size})

                if bLinear == False:
                    # 用训练得到的参数预测

                    # tl.files.assign_params(sess, load_params, network)
                    assign_op = [self.user_lf_mat.assign(self.final_user_lf_mat),
                                 self.item_lf_mat.assign(self.final_item_lf_mat),
                                 self.total_bias.assign(self.final_total_bias),
                                 self.user_bias.assign(self.final_user_bias),
                                 self.item_bias.assign(self.final_item_bias)
                                 ]

                else:  # linear model
                    # 矩阵置0.只用bias项预测
                    assign_op = [self.user_lf_mat.assign(tf.zeros(tf.shape(self.user_lf_mat))),
                                 self.item_lf_mat.assign(tf.zeros(tf.shape(self.item_lf_mat)))
                                 ]
                    self._assign_linear_var(sess)

                sess.run(assign_op)

                if TestinTrainset:
                    # batchs = self.data.generate_batch(1929285,6)
                    loss = 0
                    steps = 0
                    #               for i in range(steps):
                    #                    (batch_labels, batch_word_data, batch_doc, batch_users, batch_items, batch_ratings,
                    #                _, _)   = next(batchs)
                    for (batch_users, batch_items, batch_ratings,
                         _, _) in self.data.generate_batch_NOreviewdata(10000, 6):
                        feed_dict = {self.inputUserid: batch_users,
                                     self.inputItemid: batch_items,
                                     # self.inputContextWordid : batch_word_data,
                                     # self.label_wordid : batch_ratings,
                                     self.label_rating: batch_ratings,
                                     self.inv_weight: batch_inv_weight,
                                     self.para_lambda: 0.0
                                     }

                        loss_es, predictRating = sess.run([cost, self.predictRating], feed_dict=feed_dict)
                        # print(loss_es,predictRating)
                        loss += loss_es
                        steps += 1
                    loss = loss / steps
                    print('steps:', steps)
                else:
                   
                    dp_dict = dict_to_one(self.network.all_drop)
                    feed_dict = {self.inputUserid: data.getCVData('user'),
                                 self.inputItemid: data.getCVData('item'),
                                 # self.inputContextWordid : batch_word_data,
                                 # self.label_wordid : batch_ratings,
                                 self.label_rating: np.reshape(np.array(data.getCVData('rating')), (-1, 1)),
                                 self.inv_weight: [1.0 for _ in range(data.getCVNum())],
                                 self.para_lambda: 0.0
                                 # self.para_uv_latentfactor_size : uv_latentfactor_size
                                 }
                    
                    feed_dict.update(dp_dict)
                    loss, predict = sess.run([cost, self.predictRating], feed_dict=feed_dict)
                    print(predict)

                print("   test loss: %f" % loss)
                # print("   test acc: %f" % sess.run(acc, feed_dict=feed_dict))

        else:

            feed_dict = {self.inputUserid: data.getCVData('user'),
                         self.inputItemid: data.getCVData('item'),
                         # self.inputContextWordid : batch_word_data,
                         # self.label_wordid : batch_ratings,
                         self.label_rating: np.reshape(np.array(data.getCVData('rating')), (-1, 1)),
                         self.para_lambda: 0.0,
                         #self.para_uv_latentfactor_size: self.uv_latentfactor_size
                         }
            loss = sess.run(cost, feed_dict=feed_dict)
            print("   test loss: %f" % loss)

        return


# %%
# endregion

m = MFModel(data)



# =============================================================================
#     超参数优化
# =============================================================================



# TODO:初始化的方法也可以作为一种超参数




from skopt import gp_minimize
from skopt.plots import plot_convergence
import csv

def save_result(li,filename='MFModel_hyperparam'):
    with open(filename + '.json', 'a') as outfile:
        json.dump(li, outfile)
        outfile.write('\n') #每个词典一行。

def load_result(filename='MFModel_hyperparam'):
    x0=[]
    y0=[]
    with open(filename + '.json', 'r') as f:
        for i, line in enumerate(f):
            l = (json.loads(line))
            # l = list(dic.values())

            x0.append(l[:-1])
            y0.append(l[-1])

    if x0 == []:
        x0 = None
        y0 = None
    return x0, y0

def f(params):
    #params=(200,100,1,0.1)
    params= (128,20, 0.02, 0.005)

    uv_latentfactor_size, batch_size_s, para_lambda, para_learning_rate = params

    print('本轮参数：',params)
    # n_epoch =10*n_epoch_s
    batch_size =1000*batch_size_s

    tf.reset_default_graph()
    tl.layers.clear_layers_name()
    m.buildNetWork(uv_latentfactor_size=uv_latentfactor_size, activateFunction=None)

    m.fit(ContinueTraining=False,
          #TrainBiasOnly=False,
          batch_size=batch_size,
          context_window=4,
          n_epoch=10000,
          para_lambda=para_lambda,
          # uv_latentfactor_size=uv_latentfactor_size,
          para_learning_rate=para_learning_rate,
          tensorboard=False,
          plotResults=False,
          AutoConvergence = True,
          valandsave_epoch_freq = 10,
          optimizer = tf.train.GradientDescentOptimizer,
          lr_decay=False,
          )

    z = 0.0
    r = [int(uv_latentfactor_size),
         int(batch_size_s),
         int(para_lambda),
         float(para_learning_rate),
         float(m.final_val_cost_min)
         ]

    save_result(r)

    r.extend([m.exitcode,
              float(m.final_val_cost),
              m.training_trainlosses,
              m.training_vallosses
              ])

    save_result(r,'MFModel_trainingCurve')

    print('本轮结果：', m.final_val_cost_min)
    return m.final_val_cost_min





plt.figure("hist")

n, bins, patches = plt.hist(np.reshape(m.final_user_lf_mat,[-1]), bins=256, normed=1,edgecolor='None',facecolor='red')
plt.show()








space = [(1,512),(1, 1000), (0.0, 30.0), (0.0001, 0.7)]
x0 , y0 = load_result()

res = gp_minimize(f,  # the function to minimize
                  space,  # the bounds on each dimension of x
                  acq_func="EI",  # the acquisition function
                  n_calls=50,  # the number of evaluations of f
                  n_random_starts=10,  # the number of random initialization points
                  x0=x0,
                  y0=y0
                  # noise=0.1**2,       # the noise level (optional)
                  # random_state=123   # the random seed
                  )

print(res.x_iters,res.func_vals)
print('opt x:',res.x)
print('best score=%.4f',res.fun)
plot_convergence(res)
# endregion


