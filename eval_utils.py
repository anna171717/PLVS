# encoding=utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import sys
# reload(sys)
# sys.setdefaultencoding('utf8')

import os
import json
import hashlib
import pandas as pd
import time
import logging

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader

import misc.utils as utils
# from vist_eval.album_eval import AlbumEvaluator
from vist_eval_master.album_eval import AlbumEvaluator
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Evaluator:

    def __init__(self, save_dir, mode='val', vt_mode=1, beam_size=3):
        self.vt_mode = vt_mode
        # ref_json_path = './vist_reference/{}_reference_m{}.json'.format(mode, self.vt_mode)
        ref_json_path = '.save/simple_vist_reference/{}_reference_m{}.json'.format(mode, self.vt_mode)
        self.save_dir = save_dir
        self.beam_size = beam_size
        self.predictions = {}  # 初始化为空字典
        self.metrics = {}
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # self.reference = json.load(open(ref_json_path))
        self.reference = json.load(open("/home/ylwang/namespace/znq/CM/CM/save/simple_vist_reference/test_reference_m1.json"))
        # print("\nloading file {}".format(ref_json_path))

        self.prediction_file = os.path.join(self.save_dir, 'prediction_{}'.format(mode))
        print('*********************self.prediction_file***************************', self.prediction_file, '\n')

        self.eval = AlbumEvaluator()

    def measure(self,predictions):
        # # test on first batch
        # tmpkeys = list(self.predictions.keys())
        # # tmpkeys = ['666831', '72157631530976322']
        # tmpref = {}
        # tmppre = {}
        # for k in tmpkeys:
        #     tmpref[k] = self.reference[k]
        #
        #     tmppre[k] = self.predictions.get(k,'key not found')
        # self.reference = tmpref
        # self.predictions = tmppre

        # album eval
        self.eval.evaluate(self.reference, predictions)

        return self.eval.eval_overall

    def eval_story(self, model, topicModel, crit, dataset, loader, iteration, ifTest=True):
        # Make sure in the evaluation mode
        print("Evaluating...")
        print('*' * 20 + 'znq15')
        start = time.time()
        print('*' * 20 + 'znq16')
        model.eval()
        print('*' * 20 + 'znq17')
        if ifTest:
            dataset.test()
        else:
            dataset.val()
        print('*' * 20 + 'znq18')
        loss_sum = 0
        loss_evals = 0
        print('*' * 20 + 'znq19')
        predictions = {}
        self.prediction_file = os.path.join(self.save_dir, 'prediction_val_{}'.format(iteration))
        print('*' * 20 + 'znq20')
        # open the file to store the predictions
        count = 0
        print('*' * 20 + 'znq21')
        for iter, batch in enumerate(loader):
            iter_start = time.time()

            feature_fc = Variable(batch['feature_fc']).cuda()
            target = Variable(batch['split_story']).cuda()
            # keywords = Variable(batch['keywords']).cuda()

            conv_feature = Variable(batch['feature_conv']).cuda() if 'feature_conv' in batch else None
            count += feature_fc.size(0)

            with torch.no_grad():
                keywords = topicModel(feature_fc)

            count += feature_fc.size(0)
            with torch.no_grad():  # gjj weap model
                output = model(feature_fc, keywords, target)

            loss = crit(output, target).item()  # .data[0] --> .item() gjj-2
            loss_sum += loss
            loss_evals += 1

            # forward the model to also get generated samples for each video
            results, _ = model.predict(feature_fc, keywords, beam_size=self.beam_size)

            # sets = decode_story(dataset.get_vocab(), results)
            sents = utils.decode_story(dataset.get_vocab(), results)
            indexes = batch['index'].numpy()
            # print(f"======{len(indexes)}")
            for j, story in enumerate(sents):
                if self.vt_mode == 1:  # 1: album_id, 2: joined flickr_id
                    id = dataset.get_aid(indexes[j])
                else:
                    id = dataset.get_fid(indexes[j])
                if id not in predictions:
                    predictions[id] = [story]

            print("Evaluate iter {}/{}  {:04.2f}%. Time used: {}".format(iter,
                                                                         len(loader),
                                                                         iter * 100.0 / len(loader),
                                                                         time.time() - iter_start))

        print('*' * 20 + 'znq22')
        metrics = self.measure(predictions)  # compute all the language metrics
        print('*' * 20 + 'znq23')
        # write to json
        json_prediction_file = '{}.json'.format(self.prediction_file)
        with open(json_prediction_file, 'w') as f:
            json.dump(predictions, f)
        print('*' * 20 + 'znq24')
        # Switch back to training mode
        model.train()
        dataset.train()
        print("Evaluation finished. Evaluated {} samples. Time used: {}".format(count, time.time() - start))
        return loss_sum / loss_evals, predictions, metrics

    def test_story(self, model: object, dataset: object, loader: object, hamming_diversity: object,
                   hamming_f: object, hamming_n: object) -> object:
        print('*' * 20 + 'znq25')
        # filename
        if hamming_diversity:
            self.prediction_file = os.path.join(self.save_dir,
                                                'prediction_test_hamming_n{}_f{}'.format(hamming_n, hamming_f))
            self.prediction_score_file = os.path.join(self.save_dir,
                                                      'test_scores_hamming_n{}_f{}.json'.format(hamming_n, hamming_f))
        else:
            self.prediction_file = os.path.join(self.save_dir, 'prediction_test_nohamming')
            self.prediction_score_file = os.path.join(self.save_dir, 'test_scores_nohamming.json')
        print('*' * 20 + 'znq26')
        print("Evaluating...")
        start = time.time()
        model.eval()
        dataset.test()

        predictions = {}
        # keywords = {}

        # print one story's BS process
        # batch = dataset.get_by_fid('5694501782_5693928729_5694502126_5693929211_5694502472')
        # feature_fc = torch.tensor(batch['feature_fc']).unsqueeze(0).cuda()
        # keywords = topicModel(feature_fc)
        # results, _ = model.predict(feature_fc, keywords, beam_size=self.beam_size)
        # sents = utils.decode_story(dataset.get_vocab(), results)
        # print()
        # print(sents)
        # return
        print('*' * 20 + 'znq27')
        for iter, batch in enumerate(loader):
            iter_start = time.time()

            feature_fc = Variable(batch['feature_fc']).cuda()
            plans = batch['plans']#5，5，128
            # keywords = topicModel(feature_fc)
            keywords = plans.to(device)

            results, _ = model.predict(feature_fc, keywords, beam_size=self.beam_size)

            # sents = utils.decode_story_with_keywords(dataset.get_vocab(), results, results_kwords)
            sents = utils.decode_story(dataset.get_vocab(), results)
            print('*' * 20 + 'znq28')
            indexes = batch['index'].numpy()
            # for j, (story, kwords) in enumerate(sents):
            for j, story in enumerate(sents):
                if self.vt_mode == 1:
                    id = dataset.get_aid(indexes[j])
                else:
                    id = dataset.get_fid(indexes[j])
                if id not in predictions:
                    predictions[id] = [story]

            print("Evaluate iter {}/{}  {:04.2f}%. Time used: {}".format(iter,
                                                                         len(loader),
                                                                         iter * 100.0 / len(loader),
                                                                         time.time() - iter_start))
        print('*' * 20 + 'znq29')
        metrics = self.measure(predictions)  # compute all the language metrics
        print('*' * 20 + 'znq30')
        # write to json
        # for id in predictions:
        #     predictions[id].insert(0, keywords[id])
        print('*' * 20 + 'znq31')
        json_prediction_file = '{}.json'.format(self.prediction_file)
        with open(json_prediction_file, 'w') as f:
            json.dump(predictions, f)
        print('*' * 20 + 'znq32')
        json.dump(metrics, open(self.prediction_score_file, 'w'))
        # Switch back to training mode
        print('*' * 20 + 'znq33')
        print("Test finished. Time used: {}".format(time.time() - start))
        return predictions, metrics
        # return predictions

# def decode_story(id2word, result):
#     """
#     :param id2word: vocab
#     :param result: (batch_size, story_size, seq_length)
#     :return:
#     out: a list of stories. the size of the list is batch_size
#     """
#     batch_size, story_size, seq_length = result.size()
#     out = []
#     for i in range(batch_size):
#         txt = ''
#         for j in range(story_size):
#             for k in range(seq_length):
#                 vocab_id = result[i, j, k]
#                 if vocab_id > 0:
#                     txt = txt + ' ' + id2word[str(vocab_id.cpu().numpy())]
#                 else:
#                     break
#         out.append(txt)
#     return out