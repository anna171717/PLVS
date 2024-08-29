import sys
import json
import h5py
import os
import os.path
import numpy as np
import random
import csv
import spacy
import re
from PIL import Image

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision import models as pre_trained_models
from transformers import BertTokenizer, BertModel
# import misc.utils as utils

path_story_h5 = "/home/ylwang/namespace/znq/data/VIST/story.h5"
path_story_line_json = "/home/ylwang/namespace/znq/data/plans/plan_story_line .json"
# path_story_line_json ="/home/ylwang/namespace/znq/data/VIST/simple_plan_storyline.json"
path_ref_dir = './vist_reference'
# output_file_dir = "/home/ylwang/namespace/znq/data/plans/"
output_file_dir = "/home/ylwang/namespace/znq/data/plans/plan/"
# output_file_dir = "/home/ylwang/namespace/znq/data/save/"
# path_img_dir = "/home/ylwang/namespace/znq/data/new/images/"
path_img_dir = "/home/ylwang/newspace2/znq/data/image/img/"
# path_img_dir = "/home/ylwang/namespace/znq/data/VIST/simple_imge/"
path_resnet_features = "/home/ylwang/namespace/znq/data/resnet_features/"
# path_resnet_features = "/home/ylwang/newspace2/znq/data/resnet_features/"
path_concept_pr = "/home/ylwang/namespace/znq/CoVS_monkey/concept/pr/"
path_concept_co = "/home/ylwang/namespace/znq/CoVS_monkey/concept/clarifai/"
path_save_concept_pr = "/home/ylwang/namespace/znq/data/concept/"
feat_size = 2048

preprocess = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class VISTDataset(Dataset):
    def __init__(self):
        self.mode = 'train'

        # open the hdf5 file
        print('DataLoader loading story h5 file: ', path_story_h5)
        self.story_h5 = h5py.File(path_story_h5, 'r', driver='core')['story']
        print("story's max sentence length is ", self.story_h5.shape[1])

        print('DataLoader loading story_line json file: ', path_story_line_json)
        self.story_line = json.load(open(path_story_line_json))

        self.id2word = self.story_line['id2words']
        self.word2id = self.story_line['words2id']
        self.vocab_size = len(self.id2word)
        print('vocab size is ', self.vocab_size)
        self.story_ids = {'train': [], 'val': [], 'test': []}
        self.story_ids['train'] = list(self.story_line['train'].keys())
        self.story_ids['val'] = list(self.story_line['val'].keys())
        self.story_ids['test'] = list(self.story_line['test'].keys())

        print('There are {} training data, {} validation data, and {} test data'.format(len(self.story_ids['train']),
                                                                                        len(self.story_ids['val']),
                                                                                        len(self.story_ids['test'])))
        # self.embedding = BertTokenizer.from_pretrained("./bert_localpath/")
        # self.embedding = BertTokenizer.from_pretrained("bert-base-uncased")
        self.embedding = BertTokenizer.from_pretrained("/home/ylwang/namespace/znq/CoVS_monkey/Bert/")

        self.bert_length = 128

        # write reference files for storytelling
        if not os.path.exists(path_ref_dir):
            os.makedirs(path_ref_dir)

        # mode 1
        for split in ['train', 'val', 'test']:
            reference = {}
            for story in self.story_line[split].values():
                if story['album_id'] not in reference:
                    reference[story['album_id']] = [story['origin_text']]
                else:
                    reference[story['album_id']].append(story['origin_text'])
            with open(os.path.join(path_ref_dir, '{}_reference_m1.json'.format(split)), 'w') as f:
                json.dump(reference, f)
        # mode 2
        for split in ['train', 'val', 'test']:
            reference = {}
            for story in self.story_line[split].values():
                fid = '_'.join(story['flickr_id'])
                if fid not in reference:
                    reference[fid] = [story['origin_text']]
                else:
                    reference[fid].append(story['origin_text'])
            if split == 'train':
                self.train_reference = reference
            with open(os.path.join(path_ref_dir, '{}_reference_m2.json'.format(split)), 'w') as f:
                json.dump(reference, f)

        ## concept
        ##concept_pr
        self.concept= {}
        self.not_find_photo = []
        # data = { }
        #
        # for split in ['train', 'val']:
        #     new_dict = { }
        #     json_file_path = os.path.join(path_concept_pr, f'VIST_{split}_diverse.json')
        #     with open(json_file_path, 'r') as f:
        #         data[split] = json.load(f)
        #         for sublist in data[split]:
        #             for dictionary in sublist:
        #                 # for key, value in dictionary.items():
        #                 #     print(f"Key: {key}, Value: {value}")
        #                             # for item in data:
        #                 photo_id = dictionary["photo_flickr_id"]
        #                 # print(type(dictionary))
        #                 # print(dictionary)
        #                 Noun = dictionary.get("Noun", [])
        #                 Word = dictionary.get("Word", [])
        #                 # self.concept[split][new_dict] = {"Noun": Noun, "Word": Word}
        #                 Noun.extend(Word)
        #                 new_dict[photo_id] = Noun
        #     self.concept[split]=new_dict
        #     print("concept", self.concept[split])
        #
        # split = 'test'
        # test_dict = {}
        # json_file_path = os.path.join(path_concept_pr, f'VIST_{split}_diverse.json')
        # with open(json_file_path, 'r') as f:
        #     data[split] = json.load(f)
        #     for dictionary in data[split]:
        #             # for key, value in dictionary.items():
        #             #     print(f"Key: {key}, Value: {value}")
        #             # for item in data:
        #             photo_id = dictionary["photo_flickr_id"]
        #             print(type(dictionary))
        #             print(dictionary)
        #             Noun = dictionary.get("Noun", [])
        #             Word = dictionary.get("Word", [])
        #             # self.concept[split][new_dict] = {"Noun": Noun, "Word": Word}
        #             Noun.extend(Word)
        #             test_dict[photo_id] = Noun
        # self.concept[split] = new_dict
        # print("concept", self.concept[split])
        # for split in ['train', 'val', 'test']:
        #     with open(os.path.join(path_save_concept_pr, f'{split}_concept_keywords.json'), 'w') as f:
        #             json.dump(self.concept[split], f)
        #     max_key_length = max(len(key) for key in self.concept[split].keys())
        # self.max_concept_length=max_key_length
        for split in ['train', 'val', 'test']:
            # 11维度
            json_file_path = os.path.join(path_save_concept_pr, f'{split}_concept_keywords.json')
            with open(json_file_path, 'r') as f:
                self.concept[split] = json.load(f)
            max_key_length = max(len(key) for key in self.concept[split].keys())
            self.max_concept_length = max_key_length
            # print("vqa_plan", self.VQA_plan[split])

        # # monkey
        # # generate keywords
        # stop_word_file = os.path.join(path_ref_dir, 'stopwords_lda.txt')
        print('generate plan')
        # self.init_VQA_plan()
        # print('load cat plan')
        # load onedim VQA Plan
        self.VQA_plan = {}

        for split in ['train', 'val', 'test']:
            # 11维度
            json_file_path = os.path.join(output_file_dir, f'{split}_vqa_plan.json')
            # json_file_path = os.path.join(output_file_dir, f'{split}_simple_vqa_plan.json')
            with open(json_file_path, 'r') as f:
                self.VQA_plan[split] = json.load(f)
            # print("vqa_plan", self.VQA_plan[split])
            # self.plan_number = int(
            #     sum(len(value) for value in self.VQA_plan[split].values()) / len(self.VQA_plan[split].keys()))

        # load id2pos
        self.nlp = spacy.load('en_core_web_sm')
        self.pos_set = ["$", "``", "''", ",", "-LRB-", "-RRB-", ".", ":", "ADD", "AFX", "CC", "CD", "DT", "EX", "FW",
                        "GW", "HYPH", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NFP", "NIL", "NN", "NNP", "NNPS", "NNS",
                        "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SP", "SYM", "TO", "UH", "VB", "VBD",
                        "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB", "XX", "_SP"]
        self.pos_hamming = [""''"", "', '", "-", "-LRB-", "-RRB-", "ADD", "AFX", "GW", "HYPH", "SP", "SYM", "XX", "_SP",
                            "NN", "DT", ".,", ".", "IN", "VBD", "JJ", "NNS", "PRP", "RB", "VB", ",", ", ", "TO", "CC",
                            "VBG", "PRP$", "VBZ", "VBN", "VBP", "RP", "MD", "CD", "EX", "WRB", "POS", "WP", "WDT",
                            "JJS", "JJR", "NNP", "PDT", "RBR", ":", "``", "RBS", "UH", ")", "FW", "(", "$", "''", "#"]
        # load id2pos
        print('load id2pos')
        with open(os.path.join(path_ref_dir, 'id2pos.json'), "r") as id2pos_file:
            self.id2pos = json.load(id2pos_file)

        self.concept_max_length = 0
        self.plan_max_length = 0
        # # save id2pos to json
        # self.id2pos = {}
        # for i in range(self.vocab_size):
        #     word = self.id2word[str(i)]
        #     token = self.nlp(word)[0]
        #     self.id2pos[i] = self.pos_set.index(token.tag_)
        # id2pos_json = json.dumps(self.id2pos)
        # f = open(os.path.join(path_ref_dir, 'id2pos.json'), "w")
        # f.write(id2pos_json)
        # f.close()

    def __getitem__(self, index):
        story_id = self.story_ids[self.mode][index]
        story = self.story_line[self.mode][story_id]

        # # load img
        # imgs = torch.zeros((story['length'], 3, 224, 224), dtype = torch.float32)
        # for i in range(story['length']):
        #     path_img = os.path.join(path_img_dir, self.mode, '{}.jpg'.format(story['flickr_id'][i]))
        #     img = Image.open(path_img)
        #     img = preprocess(img)
        #     imgs[i] = img
        # sample = {'imgs': imgs}

        # load feature
        feature_fc = np.zeros((story['length'], feat_size), dtype='float32')
        for i in range(story['length']):
            fc_path = os.path.join(path_resnet_features, 'fc', self.mode, '{}.npy'.format(story['flickr_id'][i]))
            feature_fc[i] = np.load(fc_path)
        sample = {'feature_fc': feature_fc}
        # sample['feature_fc'] = feature_fc
        # load story
        split_story = self.story_h5[story['text_index']]  # split story - 5*30; whole story - 1*?
        sample['split_story'] = np.int64(split_story)

        sample['index'] = np.int64(index)
        fid = '_'.join(story['flickr_id'])
        sample['fid'] = fid
        for i in range(story['length']):
          image_name = story['flickr_id']
        sample['image_id']=image_name
        # self.concept_max_length = 0
        # self.plan_max_length = 0
        # load concept keyword
        # concept_embedding_list = torch.empty(story['length'],self.max_concept_length,self.bert_length)  # 5,128
        concept_bert_length=self.bert_length * 2
        concept_embedding_list = torch.empty(story['length'],  concept_bert_length)  # 5,256
        for i in range(story['length']):

            img_name = story['flickr_id'][i]
            # print(img_name)
            if self.mode in self.concept:
                if img_name not in self.concept[self.mode]:
                    self.concept[self.mode][img_name] = {}  # 创建一个新的空字典
            else:
                self.concept[self.mode] = {img_name: {}}  # 创建一个新的字典

            # 然后可以继续访问 self.concept[self.mode][img_nam
            text_list = self.concept[self.mode][img_name]
            # text_list = list(text_list)
            """用Bert编码"""
            text = ""
            text = "[SEP]".join(text_list)
            text = f"[CLS]{text.strip()}"

            concept_length = len(text)
            # 更新最大长度
            if concept_length > self.concept_max_length:
                self.concept_max_length = concept_length

            concept_embedding = self.embedding.encode_plus(
                text=text,  # the sentence to be encoded
                add_special_tokens=True,  # Add [CLS] and [SEP]
                max_length=concept_bert_length,  # maximum length of a sentence
                pad_to_max_length=True, # Add [PAD]s
                return_attention_mask=True,  # Generate the attention mask
                return_tensors='pt',  # ask the function to return PyTorch tensors
            )["input_ids"]
            """bert编码结束"""
            concept_embedding_list[i] = concept_embedding
            # tensor维度【5，bert——maxlen】
        # print("concept_length:",self.concept_max_length)
        sample['concept_max'] = self.concept_max_length
        sample['concept'] = concept_embedding_list


        # load vqa keywords
        # not_find_photo = []
        plan_bert_length=self.bert_length * 24
        plan_embedding_list = torch.empty(story['length'], plan_bert_length)  # 5,1024
        for i in range(story['length']):
            text_list = []
            img_name = story['flickr_id'][i]
            # print(img_name)
            if img_name not in self.VQA_plan[self.mode]:
                self.not_find_photo.append(img_name)
                self.VQA_plan[self.mode][img_name] = {}  # 创建一个新的空字典
            text_list = self.VQA_plan[self.mode][img_name]
            # text_list = list(text_list)
            """用Bert编码"""
            text = ""
            # #QA
            for item in text_list:
                if item != text_list[-1]:
                    text += f"question: {item['question']} [SEP] answer: {item['answer']} [SEP]"
                    # text += f"{item['question']} [SEP] {item['answer']} [SEP] "
                    # text += f"{item['answer']} [SEP] "
                else:
                    text += f"question: {item['question']} [SEP] answer: {item['answer']}"
                    # text += f"{item['question']} [SEP] {item['answer']}"
                    # text += f"{item['answer']} "
            # text = "[SEP]".join(text_list)
            text = f"[CLS]{text.strip()}"
            plan_length = len(text)
            # 更新最大长度
            if plan_length > self.plan_max_length:
                self.plan_max_length = plan_length
            plan_embedding = self.embedding.encode_plus(
                text=text,  # the sentence to be encoded
                add_special_tokens=True,  # Add [CLS] and [SEP]
                max_length=plan_bert_length,  # maximum length of a sentence
                pad_to_max_length=True,  # Add [PAD]s
                return_attention_mask=True,  # Generate the attention mask
                return_tensors='pt',  # ask the function to return PyTorch tensors
            )["input_ids"]
            """bert编码结束"""
            plan_embedding_list[i] = plan_embedding
            # tensor维度【5，bert——maxlen】
        with open(os.path.join(output_file_dir, f'plan_not_find_photo_2 .json'), 'w', encoding='utf-8') as f:
            json.dump(self.not_find_photo, f, ensure_ascii=False)
        # print("plan_length:", self.plan_max_length)
        sample['plan_max']=self.plan_max_length
        sample['plans'] = plan_embedding_list
        # print()
        # print(f'not found photo:{self.not_find_photo}')
        return sample

    def get_by_fid(self, _fid):
        counter = 0
        for index, story in enumerate(self.story_line['test'].values()):
            fid = '_'.join(story['flickr_id'])
            if fid == _fid:
                if counter >= 1:
                    # feature
                    feature_fc = np.zeros((story['length'], feat_size), dtype='float32')
                    for i in range(story['length']):
                        fc_path = os.path.join(path_resnet_features, 'fc', 'test',
                                               '{}.npy'.format(story['flickr_id'][i]))
                        feature_fc[i] = np.load(fc_path)
                    sample = {'feature_fc': feature_fc}
                    # gt
                    split_story = self.story_h5[story['text_index']]  # split story - 5*30; whole story - 1*?
                    sample['split_story'] = np.int64(split_story)
                    # load lda keywords distribution
                    sample['keywords'] = self.keywords['test'][index, :, :].astype(np.float32)

                    return sample
                else:
                    counter += 1

    def __len__(self):
        return len(self.story_ids[self.mode])

    def train(self):
        self.mode = 'train'

    def val(self):
        self.mode = 'val'

    def test(self):
        self.mode = 'test'

    def get_GT(self, index):
        """ get GT storys by batch index, used by criterion's rl reward
            train_reference is mode 2
        """

        story_id = self.story_ids[self.mode][index]
        story = self.story_line[self.mode][story_id]
        fid = '_'.join(story['flickr_id'])
        return self.train_reference[fid]

    def get_aid(self, index):
        """ get album_id by batch index
        """

        story_id = self.story_ids[self.mode][index]
        return self.story_line[self.mode][story_id]['album_id']

    def get_fid(self, index):
        """ get joint flickr_id by batch index
        """

        story_id = self.story_ids[self.mode][index]
        return '_'.join(self.story_line[self.mode][story_id]['flickr_id'])

    def get_all_id(self, index):
        story_id = self.story_ids[self.mode][index]
        return self.story_line[self.mode][story_id]['album_id'], self.story_line[self.mode][story_id]['flickr_id']

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.id2word

    def get_word2id(self):
        return self.word2id

    def get_whole_story_length(self):
        return self.full_story_h5.shape[1]

    def get_story_length(self):
        return self.story_h5.shape[1]

    def get_caption_length(self):
        return self.desc_h5.shape[1]

    def pos_ifhamming(self, index):
        """ if index in vocab is in pos hamming list
        """

        return self.pos_set[self.id2pos[str(index)]] in self.pos_hamming

    def init_VQA_plan(self, plan_keyword=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import json
        import os
        # vqa
        checkpoint = "/home/ylwang/namespace/znq/CM/CM/Monkey-main/monkey_model/arg/"
        model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map='cuda', trust_remote_code=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        tokenizer.padding_side = 'left'
        tokenizer.pad_token_id = tokenizer.eod_id
        for split in ['train', 'val', 'test']:
            img_folder = path_img_dir
            split_folder = os.path.join(img_folder, split)
            # plan = []
            questions = [
                "What is the theme of the picture",  # theme
                "What is in the picture",  # object
                "What are the objects in the picture",
                "What color is it",
                "What is the shape of the object",
                "What is the man doing in the picture",  # active
                "What happened in the picture",
                "What is the person or animal in the picture doing",
                "Where the events in the image took place",  # Localization
                "What is people going to do",  # Forecasting
                "What will happen",  # plan
            ]
            print("start")
            plans = []
            plans_cat = []
            plans_onedim = []
            VQA_plan_cat = {}
            vqa_plans = {}
            for img_name in os.listdir(split_folder):
                # plan_keywords ={ }
                VQA = []
                keyword_texts = []
                one_dim_keyword = []
                img_path = os.path.join(split_folder, img_name)
                img_name_without_extension = os.path.splitext(img_name)[0]
                plan_keyword = {
                    'id': img_name_without_extension,
                    'plan': VQA
                }
                plan_keyword_cat = {
                    'id': img_name_without_extension,
                    'plan': keyword_texts
                }
                plan_keyword_onedim = {
                    'id': img_name_without_extension,
                    'plan': one_dim_keyword
                }
                if img_name.endswith(".json"):  # Skip JSON files
                    continue
                    print("img_name", img_name)
                for question in questions:
                    responses = []
                    keyword_text = []
                    responses_dict = {}
                    for i in range(3):
                        query = f'<img>{img_path}</img> {question} Answer: '
                        input_ids = tokenizer(query, return_tensors='pt', padding='longest')
                        # input_ids = tokenizer( return_tensors='pt', padding='longest')
                        attention_mask = input_ids.attention_mask
                        input_ids = input_ids.input_ids
                        pred = model.generate(
                            input_ids=input_ids.cuda(),
                            attention_mask=attention_mask.cuda(),
                            do_sample=True,
                            num_beams=1,
                            max_new_tokens=512,
                            min_new_tokens=1,
                            length_penalty=1,
                            num_return_sequences=1,
                            output_hidden_states=True,
                            use_cache=True,
                            eos_token_id=tokenizer.eod_id,
                        )
                        response = tokenizer.decode(pred[0][input_ids.size(1):].cpu(), skip_special_tokens=True).strip()
                        responses.append(response)
                        print("response", response)
                        print("responses", responses)
                        # print("response", responses_dict)
                        # print("res",res)
                    # response = tokenizer.decode(pred[0][input_ids.size(1):].cpu(), skip_special_tokens=True).strip()
                    output_dict = {
                        "question": question,
                        "response": responses
                        # "response": res
                        #  "response": responses_dict
                    }
                    VQA.append(output_dict)
                    print("output_dict", output_dict)
                    keyword_text = [self.mkinput(question, responses)]
                    keyword_texts.append(keyword_text)
                    print("keyword_text:", keyword_text)
                    print("keyword_texts:", keyword_texts)
                    # if one_dim_keyword == []:
                    #     one_dim_keyword = keyword_text
                    # else:
                    #    one_dim_keyword= [self.mkinput(one_dim_keyword,keyword_text)]
                    one_dim_keyword = ' [SEP] '.join([' '.join(trip) for trip in keyword_texts])
                    print("one:", one_dim_keyword)
                    # keyword_texts =[self.mkinput(keyword_text,keyword_texts)]

                plan_keyword["id"] = img_name_without_extension
                plan_keyword['plan'] = VQA
                plans.append(plan_keyword)
                print("plans", plans)
                plan_keyword_cat["id"] = img_name_without_extension
                plan_keyword_cat["plan"] = keyword_texts
                plans_cat.append(plan_keyword_cat)
                VQA_plan_cat[img_name_without_extension] = keyword_texts
                plan_keyword_onedim["id"] = img_name_without_extension
                plan_keyword_onedim["plan"] = one_dim_keyword
                plans_onedim.append(plan_keyword_onedim)
                vqa_plans[img_name_without_extension] = one_dim_keyword
                print("vqa_plans", vqa_plans)
            output_file_dir = "/home/ylwang/namespace/znq/CM/CM/save"
            print("victory")
            # 原始数据
            with open(os.path.join(output_file_dir, f'{split}_plan_keywords_new.json'), 'w') as f:
                json.dump(plans, f)
            with open(os.path.join(output_file_dir, 'keywords_' + str(split), '{}_plan_keywords_new.npy'.format(split)),
                      'wb') as f:
                np.save(f, np.array(plans))
            print("plans", np.array(plans).shape)
            # 11维
            with open(os.path.join(output_file_dir, f'{split}_plan_keywords_cats.json'), 'w') as f:
                json.dump(plans_cat, f)
            with open(
                    os.path.join(output_file_dir, 'keywords_' + str(split), '{}_plan_keywords_cats.npy'.format(split)),
                    'wb') as f:
                np.save(f, np.array(plans_cat))
            print("plans_cat", np.array(plans_cat).shape)
            # VQA_plan_cat
            with open(os.path.join(output_file_dir, f'{split}_plan_keywords_cats.json'), 'w') as f:
                json.dump(VQA_plan_cat, f)
            with open(
                    os.path.join(output_file_dir, 'keywords_' + str(split), '{}_plan_keywords_cats.npy'.format(split)),
                    'wb') as f:
                np.save(f, np.array(VQA_plan_cat))
            print("plans_cat", np.array(plans_cat).shape)
            # 一维
            with open(os.path.join(output_file_dir, f'{split}_plan_keywords_onedim.json'), 'w') as f:
                json.dump(plans_onedim, f)
            with open(os.path.join(output_file_dir, 'keywords_' + str(split),
                                   '{}_plan_keywords_onedim.npy'.format(split)),
                      'wb') as f:
                np.save(f, np.array(plans_onedim))
            print("plans_onedims", np.array(plans_onedim).shape)
            # vqa plan 传过去的数据
            with open(os.path.join(output_file_dir, f'{split}_vqa_plan.json'), 'w') as f:
                json.dump(vqa_plans, f)
            with open(os.path.join(output_file_dir, 'keywords_' + str(split), '{}_vqa_plan.npy'.format(split)),
                      'wb') as f:
                np.save(f, np.array(vqa_plans))
            max_length = max(len(str(key)) + len(str(value)) for key, value in vqa_plans.items())
            print("max_length:", max_length)

    def get_ITFweights(self, target, param):
        pass
    # def get_ITFweights(self, target, param):
    #     # 计算权重的逻辑
    #     weights = []
    #     for t in target:
    #       # 在这里根据目标计算相应的权重值
    #        weight =1
    #        weights.append(weight)
    #     return weights
### preprocess ###
# remove album that has empty img in story_line
def vist_preprocess():
    # open the hdf5 file
    print('DataLoader loading story h5 file: ', path_story_h5)
    story_h5 = h5py.File(path_story_h5, 'r', driver='core')['story']
    print("story's max sentence length is ", story_h5.shape[1])

    print('DataLoader loading story_line json file: ', path_story_line_json)
    story_line = json.load(open(path_story_line_json))

    id2word = story_line['id2words']
    word2id = story_line['words2id']
    vocab_size = len(id2word)
    print('vocab size is ', vocab_size)
    story_ids = {'train': [], 'val': [], 'test': []}
    description_ids = {'train': [], 'val': [], 'test': []}

    story_ids['train'] = list(story_line['train'].keys())
    story_ids['val'] = list(story_line['val'].keys())
    story_ids['test'] = list(story_line['test'].keys())
    description_ids['train'] = list(story_line['image2caption']['train'].keys())
    description_ids['val'] = list(story_line['image2caption']['val'].keys())
    description_ids['test'] = list(story_line['image2caption']['test'].keys())

    print('There are {} training data, {} validation data, and {} test data'.format(len(story_ids['train']),
                                                                                    len(story_ids['val']),
                                                                                    len(story_ids['test'])))

    # get non-RGB jpgs
    # outers = {}
    for mode in ['val', 'test', 'train']:
        for s_id in story_ids[mode]:
            story = story_line[mode][s_id]

            for i in range(story['length']):
                path_img = os.path.join(path_img_dir, mode, '{}.jpg'.format(story['flickr_id'][i]))

                # image not exist
                if not os.path.isfile(path_img):
                    del story_line[mode][s_id]
                    break

                img = Image.open(path_img)

                # image not RGB
                if img.mode != 'RGB':
                    # print(story['flickr_id'], story['flickr_id'][i])
                    del story_line[mode][s_id]
                    # outers[s_id] = img.mode
                    break

    # with open(os.path.join('not_RGB_images.json'), 'w') as f:
    #     json.dump(outers, f)

    # with open('./datasets/VIST/story_line.json', 'w') as f:
    #     json.dump(story_line, f)

    id2word = story_line['id2words']
    word2id = story_line['words2id']
    vocab_size = len(id2word)
    print('vocab size is ', vocab_size)
    story_ids = {'train': [], 'val': [], 'test': []}
    description_ids = {'train': [], 'val': [], 'test': []}

    story_ids['train'] = list(story_line['train'].keys())
    story_ids['val'] = list(story_line['val'].keys())
    story_ids['test'] = list(story_line['test'].keys())
    description_ids['train'] = list(story_line['image2caption']['train'].keys())
    description_ids['val'] = list(story_line['image2caption']['val'].keys())
    description_ids['test'] = list(story_line['image2caption']['test'].keys())

    print('There are {} training data, {} validation data, and {} test data'.format(len(story_ids['train']),
                                                                                    len(story_ids['val']),
                                                                                    len(story_ids['test'])))


### resnet152 fc features ###
def resfc():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    resnet = pre_trained_models.resnet152(pretrained=True)

    # remove fc
    resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))

    resnet = resnet.to(device)
    resnet.eval()

    # open the hdf5 file
    print('DataLoader loading story h5 file: ', path_story_h5)
    story_h5 = h5py.File(path_story_h5, 'r', driver='core')['story']
    print("story's max sentence length is ", story_h5.shape[1])

    print('DataLoader loading story_line json file: ', path_story_line_json)
    story_line = json.load(open(path_story_line_json))

    id2word = story_line['id2words']
    word2id = story_line['words2id']
    vocab_size = len(id2word)
    print('vocab size is ', vocab_size)
    story_ids = {'train': [], 'val': [], 'test': []}
    description_ids = {'train': [], 'val': [], 'test': []}

    story_ids['train'] = list(story_line['train'].keys())
    story_ids['val'] = list(story_line['val'].keys())
    story_ids['test'] = list(story_line['test'].keys())
    description_ids['train'] = list(story_line['image2caption']['train'].keys())
    description_ids['val'] = list(story_line['image2caption']['val'].keys())
    description_ids['test'] = list(story_line['image2caption']['test'].keys())

    print('There are {} training data, {} validation data, and {} test data'.format(len(story_ids['train']),
                                                                                    len(story_ids['val']),
                                                                                    len(story_ids['test'])))

    for mode in ['test', 'train']:
        print(mode)
        for s_id in story_ids[mode]:
            story = story_line[mode][s_id]

            for i in range(story['length']):
                path_img = os.path.join(path_img_dir, mode, '{}.jpg'.format(story['flickr_id'][i]))
                img = Image.open(path_img)
                img = preprocess(img).unsqueeze(0)

                img = img.to(device)
                feature_fc = resnet(img).squeeze()
                feature_fc = np.array(feature_fc.cpu().detach())
                np.save(os.path.join(path_resnet_features, 'fc', mode, '{}.npy'.format(story['flickr_id'][i])),
                        feature_fc)

###
# resfc()

# VISTDataset(num_topics = 16)