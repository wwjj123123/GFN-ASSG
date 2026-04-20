'''
Description: 
version: 
Author: chenhao
Date: 2021-06-09 14:17:37
'''
import os
import sys
import torch.nn.functional as F
import torch
sys.path.append(r'./LAL-Parser/src_joint')
import re
import json
import pickle
import random
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
from torch.utils.data import Dataset
from tree import calculate_shortest_paths
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from distance_based_weighted_matrix import aspect_oriented_tree


def ParseData(data_path):
    with open(data_path) as infile:
        all_data = []
        data = json.load(infile)
        for d in data:
            for aspect in d['aspects']:
                text_list = list(d['token'])
                tok = list(d['token'])       # word token
                length = len(tok)            # real length
                # if args.lower == True:
                tok = [t.lower() for t in tok]
                tok = ' '.join(tok)
                asp = list(aspect['term'])   # aspect
                asp = [a.lower() for a in asp]
                asp = ' '.join(asp)
                label = aspect['polarity']   # label
                pos = list(d['pos'])         # pos_tag 
                head = list(d['head'])       # head
                deprel = list(d['deprel'])   # deprel
                # position
                aspect_post = [aspect['from'], aspect['to']] 
                post = [i-aspect['from'] for i in range(aspect['from'])] \
                       +[0 for _ in range(aspect['from'], aspect['to'])] \
                       +[i-aspect['to']+1 for i in range(aspect['to'], length)]
                # aspect mask
                if len(asp) == 0:
                    mask = [1 for _ in range(length)]    # for rest16
                else:
                    mask = [0 for _ in range(aspect['from'])] \
                       +[1 for _ in range(aspect['from'], aspect['to'])] \
                       +[0 for _ in range(aspect['to'], length)]
                
                sample = {'text': tok, 'aspect': asp, 'pos': pos, 'post': post, 'head': head,\
                          'deprel': deprel, 'length': length, 'label': label, 'mask': mask, \
                          'aspect_post': aspect_post, 'text_list': text_list}
                all_data.append(sample)

    return all_data


def build_tokenizer(fnames, max_length, data_file):
    parse = ParseData
    if os.path.exists(data_file):
        print('loading tokenizer:', data_file)
        tokenizer = pickle.load(open(data_file, 'rb'))
    else:
        tokenizer = Tokenizer.from_files(fnames=fnames, max_length=max_length, parse=parse)
        pickle.dump(tokenizer, open(data_file, 'wb'))
    return tokenizer


class Vocab(object):
    ''' vocabulary of dataset '''
    def __init__(self, vocab_list, add_pad, add_unk):
        self._vocab_dict = dict()
        self._reverse_vocab_dict = dict()
        self._length = 0
        if add_pad:
            self.pad_word = '<pad>'
            self.pad_id = self._length
            self._length += 1
            self._vocab_dict[self.pad_word] = self.pad_id
        if add_unk:
            self.unk_word = '<unk>'
            self.unk_id = self._length
            self._length += 1
            self._vocab_dict[self.unk_word] = self.unk_id
        for w in vocab_list:
            self._vocab_dict[w] = self._length
            self._length += 1
        for w, i in self._vocab_dict.items():   
            self._reverse_vocab_dict[i] = w  
    
    def word_to_id(self, word):  
        if hasattr(self, 'unk_id'):
            return self._vocab_dict.get(word, self.unk_id)
        return self._vocab_dict[word]
    
    def id_to_word(self, id_):   
        if hasattr(self, 'unk_word'):
            return self._reverse_vocab_dict.get(id_, self.unk_word)
        return self._reverse_vocab_dict[id_]
    
    def has_word(self, word):
        return word in self._vocab_dict
    
    def __len__(self):
        return self._length
    
    @staticmethod
    def load_vocab(vocab_path: str):
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)


class Tokenizer(object):
    ''' transform text to indices '''
    def __init__(self, vocab, max_length, lower, pos_char_to_int, pos_int_to_char):
        self.vocab = vocab
        self.max_length = max_length
        self.lower = lower

        self.pos_char_to_int = pos_char_to_int
        self.pos_int_to_char = pos_int_to_char
    
    @classmethod
    def from_files(cls, fnames, max_length, parse, lower=True):
        corpus = set()
        pos_char_to_int, pos_int_to_char = {}, {}
        for fname in fnames:
            for obj in parse(fname):
                text_raw = obj['text']
                if lower:
                    text_raw = text_raw.lower()
                corpus.update(Tokenizer.split_text(text_raw)) 
        return cls(vocab=Vocab(corpus, add_pad=True, add_unk=True), max_length=max_length, lower=lower, pos_char_to_int=pos_char_to_int, pos_int_to_char=pos_int_to_char)
    
    @staticmethod
    def pad_sequence(sequence, pad_id, maxlen, dtype='int64', padding='post', truncating='post'):
        x = (np.zeros(maxlen) + pad_id).astype(dtype)
        if truncating == 'pre':
            trunc = sequence[-maxlen:] 
        else:
            trunc = sequence[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == 'post':
            x[:len(trunc)] = trunc 
        else:
            x[-len(trunc):] = trunc
        return x
    
    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        words = Tokenizer.split_text(text)
        sequence = [self.vocab.word_to_id(w) for w in words] 
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence.reverse()  
        return Tokenizer.pad_sequence(sequence, pad_id=self.vocab.pad_id, maxlen=self.max_length, 
                                      padding=padding, truncating=truncating)
    
    @staticmethod
    def split_text(text):
        # for ch in ["\'s", "\'ve", "n\'t", "\'re", "\'m", "\'d", "\'ll", ",", ".", "!", "*", "/", "?", "(", ")", "\"", "-", ":"]:
        #     text = text.replace(ch, " "+ch+" ")
        return text.strip().split()


class SentenceDataset(Dataset):
    ''' PyTorch standard dataset class '''
    def __init__(self, fname, tokenizer, opt, vocab_help):

        parse = ParseData
        post_vocab, pos_vocab, dep_vocab, pol_vocab = vocab_help
        data = list()
        polarity_dict = {'positive':0, 'negative':1, 'neutral':2}
        for obj in tqdm(parse(fname), total=len(parse(fname)), desc="Training examples"):
            text = tokenizer.text_to_sequence(obj['text'])
            aspect = tokenizer.text_to_sequence(obj['aspect'])  # max_length=10
            
            
            
            
            # 新增二，设置与aspect word之间的距离 
            # 保存一下post的绝对值，类似：post = [-5, -4, -3, -2, -1, 0, 0, 1, 2]
            ped_post = [abs(x) for x in obj['post']]
            ped_post = tokenizer.pad_sequence(ped_post, pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post', truncating='post')
            # 方面词,用于后续建立方面词为根的树，来找其位置
            aspect_index = torch.tensor(obj['aspect_post'])
            
            
            
            
            post = [post_vocab.stoi.get(t, post_vocab.unk_index) for t in obj['post']]
            post = tokenizer.pad_sequence(post, pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post', truncating='post')
            pos = [pos_vocab.stoi.get(t, pos_vocab.unk_index) for t in obj['pos']]
            pos = tokenizer.pad_sequence(pos, pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post', truncating='post')
            deprel = [dep_vocab.stoi.get(t, dep_vocab.unk_index) for t in obj['deprel']]
            deprel = tokenizer.pad_sequence(deprel, pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post', truncating='post')
            mask = tokenizer.pad_sequence(obj['mask'], pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post', truncating='post')





            #新增一部分，在这里直接对pos词性进行筛选，并传递索引（键值）
            pos_index_vocab = {'NN':1,'JJ':2,'RB':3,'CC':4,'VBD':5,'VB':6,'VBN':7,'PRP':8,'VBG':9,'MD':10,'WDT':11,'DT':12,'RB':13}
            pos_obj_vocab = []
            for t in obj['pos']:
                if t in pos_index_vocab.keys():# pos_vocab.stoi：这通常是一个字典，其中 stoi 可能是 “string to index” 的缩写，表示将字符串映射到索引的字典。
                    for index, (key, value) in enumerate(pos_index_vocab.items()):
                        if key == t:
                            pos_obj_vocab.append(value)
                else:
                    pos_obj_vocab.append(0)
            # pos_obj_vocab = [label for t in obj['pos'] if t in pos_vocab.stoi.keys() for label, index in pos_index_vocab.items() if index == t]
            pos_obj_vocab = tokenizer.pad_sequence(pos_obj_vocab, pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post', truncating='post')
  





            adj = np.ones(opt.max_length) * opt.pad_id
            if opt.parseadj:
                from absa_parser import headparser
                # * adj
                headp, syntree = headparser.parse_heads(obj['text'])
                adj = softmax(headp[0])
                adj = np.delete(adj, 0, axis=0)
                adj = np.delete(adj, 0, axis=1)
                adj -= np.diag(np.diag(adj))
                if not opt.direct:
                    adj = adj + adj.T
                adj = adj + np.eye(adj.shape[0])
                adj = np.pad(adj, (0, opt.max_length - adj.shape[0]), 'constant')
            
            if opt.parsehead:
                from absa_parser import headparser
                headp, syntree = headparser.parse_heads(obj['text'])
                syntree2head = [[leaf.father for leaf in tree.leaves()] for tree in syntree]
                head = tokenizer.pad_sequence(syntree2head[0], pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post', truncating='post')
            else:
                head = tokenizer.pad_sequence(obj['head'], pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post', truncating='post')
            length = obj['length']
            polarity = polarity_dict[obj['label']]
            data.append({
                'text': text, 
                'aspect': aspect, 
                'post': post,
                'pos': pos,
                'deprel': deprel,
                'head': head,
                'adj': adj,
                'mask': mask,
                'length': length,
                'polarity': polarity,
                'pos_obj_vocab': pos_obj_vocab,
                'ped_post': ped_post,
                'asp_index':aspect_index
            })

        self._data = data

    def __getitem__(self, index):
        return self._data[index]
    
    def __len__(self):
        return len(self._data)

def _load_wordvec(data_path, embed_dim, vocab=None):
    with open(data_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        word_vec = dict()
        if embed_dim == 200:
            for line in f:
                tokens = line.rstrip().split()
                if tokens[0] == '<pad>' or tokens[0] == '<unk>': # avoid them
                    continue
                if vocab is None or vocab.has_word(tokens[0]):
                    word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
        elif embed_dim == 300:
            for line in f:
                tokens = line.rstrip().split()
                if tokens[0] == '<pad>': # avoid them
                    continue
                elif tokens[0] == '<unk>':
                    word_vec['<unk>'] = np.random.uniform(-0.25, 0.25, 300)
                word = ''.join((tokens[:-300]))
                if vocab is None or vocab.has_word(tokens[0]):
                    word_vec[word] = np.asarray(tokens[-300:], dtype='float32')
        else:
            print("embed_dim error!!!")
            exit()
            
        return word_vec

def build_embedding_matrix(vocab, embed_dim, data_file):
    if os.path.exists(data_file):
        print('loading embedding matrix:', data_file)
        embedding_matrix = pickle.load(open(data_file, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(vocab), embed_dim))
        fname = './glove/glove.840B.300d.txt'
        word_vec = _load_wordvec(fname, embed_dim, vocab)
        for i in range(len(vocab)):
            vec = word_vec.get(vocab.id_to_word(i))
            if vec is not None:
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(data_file, 'wb'))
    return embedding_matrix


def softmax(x):
    if len(x.shape) > 1:
        # matrix
        tmp = np.max(x, axis=1)
        x -= tmp.reshape((x.shape[0], 1))
        x = np.exp(x)
        tmp = np.sum(x, axis=1)
        x /= tmp.reshape((x.shape[0], 1))
    else:
        # vector
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp
    return x


class Tokenizer4BertGCN:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.max_seq_len = max_seq_len # 表示最大序列长度。这个参数在后续的文本处理过程中用于裁剪或填充文本
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name) # 加载指定的预训练 BERT 分词器。这个分词器包含了词汇表、分词方法、特殊标记（如 [CLS] 和 [SEP]）等，可以直接用于文本的分词与转换。
        self.cls_token_id = self.tokenizer.cls_token_id # 获取 BERT 分词器中的 [CLS]（分类）标记的 ID。[CLS] 是 BERT 模型输入序列的起始符号，通常用于序列分类任务。
        self.sep_token_id = self.tokenizer.sep_token_id # 获取 BERT 分词器中的 [SEP]（分隔）标记的 ID。[SEP] 用于标记句子边界，尤其是在 BERT 处理句子对任务时，它会用来分隔两个句子。
    # 这个方法将输入的字符串 s 分词成 BERT 模型可以理解的子词单元（tokens）,BertTokenizer 使用 WordPiece 分词算法，将文本分解成更小的词或子词单元
    def tokenize(self, s):
        return self.tokenizer.tokenize(s) 
    # 将分词后的 tokens 转换成 BERT 所使用的 ID。BERT 使用一个词汇表，其中每个单词或子词都有一个对应的整数 ID。此方法将分词结果（tokens）映射为这些 ID，便于将文本转换为模型输入。
    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)
    
    # 新增方法：将 ID 列表转换为文本 tokens
    def convert_ids_to_tokens(self, ids):
        return self.tokenizer.convert_ids_to_tokens(ids)


# bert 数据预处理
class ABSAGCNData(Dataset):
    def __init__(self, fname, tokenizer, opt):   
        self.data = []
        parse = ParseData
        polarity_dict = {'positive':0, 'negative':1, 'neutral':2}
        for obj in tqdm(parse(fname), total=len(parse(fname)), desc="Training examples"):
            polarity = polarity_dict[obj['label']]
            text = obj['text']
            term = obj['aspect']
            term_start = obj['aspect_post'][0]
            term_end = obj['aspect_post'][1]
            text_list = obj['text_list'] # == token
            term_pos = obj['pos']
            left, term, right = text_list[: term_start], text_list[term_start: term_end], text_list[term_end: ]
            left_pos, asp_pos, right_pos = term_pos[: term_start], term_pos[term_start: term_end], term_pos[term_end: ]


            from absa_parser import headparser
            headp, syntree = headparser.parse_heads(text)# 使用 headparser 解析器解析文本，获取句法树（syntree）和弧矩阵信息
                       
            
            # 对解析结果进行处理，计算原始的邻接矩阵，并去掉不必要的部分。
            ori_adj = softmax(headp[0])
            ori_adj = np.delete(ori_adj, 0, axis=0) # 删除 ori_adj 的第一行（索引为 0 的行）。axis=0 表示按行删除
            ori_adj = np.delete(ori_adj, 0, axis=1) # 删除 ori_adj 的第一列（索引为 0 的列）。axis=1 表示按列删除。
            ori_adj -= np.diag(np.diag(ori_adj)) # 对角线元素置为 0
            if not opt.direct: # 将邻接矩阵进行对称处理
                ori_adj = ori_adj + ori_adj.T
            ori_adj = ori_adj + np.eye(ori_adj.shape[0]) # 将单位矩阵添加到邻接矩阵中，以确保自连接
            # 断言检查确保文本长度与邻接矩阵的维度一致
            assert len(text_list) == ori_adj.shape[0] == ori_adj.shape[1], '{}-{}-{}'.format(len(text_list), text_list, ori_adj.shape)






            # 测试这部分问题
            # # 创建依赖无向图
            # adj_graph = np.where(ori_adj != 0, 1, 0)
            # # 计算最短路径
            # D_min = calculate_shortest_paths(adj_graph, term_start, term_end)
            
            D_min = aspect_oriented_tree(opt, token=obj['text_list'], head=obj['head'],
                                                as_start=obj['aspect_post'][0], as_end=obj['aspect_post'][1])
            

            # 建立一个方面词子词词性映射表
            pos_sub_tokens_vocab = []
            # 初始化分词列表和原始索引映射
            left_tokens, term_tokens, right_tokens = [], [], [] # term_tokens记录的是方面文本
            left_tok2ori_map, term_tok2ori_map, right_tok2ori_map = [], [], []
            # left_tok2ori_map 是左侧文本的每个 token 对应的原始词汇索引。
            # term_tok2ori_map 是方面词的每个 token 对应的原始词汇索引。
            # right_tok2ori_map 是右侧文本的每个 token 对应的原始词汇索引。

            # 对左侧文本进行分词，并记录原始索引
            for ori_i, (w, w_pos) in enumerate(zip(left, left_pos)):
                for t in tokenizer.tokenize(w):
                    left_tokens.append(t)                   # * ['expand', '##able', 'highly', 'like', '##ing']
                    left_tok2ori_map.append(ori_i)          # * [0, 0, 1, 2, 2]
                    pos_sub_tokens_vocab.append(w_pos)      # 给子词赋予词性



            asp_start = len(left_tokens)  # 记录分词后方面的起始位置
            offset = len(left) # 记录分词前方面的偏移量
            
            # 对方面文本进行分词，并记录原始索引。
            for ori_i, (w, w_pos) in enumerate(zip(term, asp_pos)):        
                for t in tokenizer.tokenize(w):
                    term_tokens.append(t)
                    # term_tok2ori_map.append(ori_i)
                    term_tok2ori_map.append(ori_i + offset)
                    pos_sub_tokens_vocab.append(w_pos)      # 给子词赋予词性


            asp_end = asp_start + len(term_tokens) # 记录方面的结束位置
            offset += len(term) # 并更新偏移量= 分词前左文本+方面词文本。
            # 对右侧文本进行分词，并记录原始索引。
            for ori_i, (w, w_pos) in enumerate(zip(right, right_pos)):
                for t in tokenizer.tokenize(w):
                    right_tokens.append(t)
                    right_tok2ori_map.append(ori_i+offset)      
                    pos_sub_tokens_vocab.append(w_pos)      # 给子词赋予词性     

            # 调整分词长度，如果分词的总长度超过最大序列长度，则逐步移除左侧或右侧的词，直到满足长度要求。
            while len(left_tokens) + len(right_tokens) > tokenizer.max_seq_len-2*len(term_tokens) - 3:
                # 如果 left_tokens 的长度大于 right_tokens，则从 left_tokens 的开头（索引为 0 的位置）移除一个 token，
                # 并且相应地在 left_tok2ori_map 中移除对应的映射
                if len(left_tokens) > len(right_tokens):
                    left_tokens.pop(0)
                    left_tok2ori_map.pop(0)
                    asp_start -=1
                    asp_end -=1
                    pos_sub_tokens_vocab.pop(0)
                # 如果 right_tokens 的长度大于或等于 left_tokens，则从 right_tokens 的末尾移除一个 token，
                # 并且在 right_tok2ori_map 中移除对应的映射。
                else:
                    right_tokens.pop()
                    right_tok2ori_map.pop()
                    pos_sub_tokens_vocab.pop()
            
            # 构建输入数据
            bert_tokens = left_tokens + term_tokens + right_tokens # 构建 BERT 输入的分词列表，
            tok2ori_map = left_tok2ori_map + term_tok2ori_map + right_tok2ori_map # 构建 BERT 输入索引映射，
            
            



            #新增一部分，在这里直接对pos词性进行筛选，并传递索引（键值）
            pos_index_vocab = {'NN':1,'JJ':2,'RB':3,'CC':4,'VBD':5,'VB':6,'VBN':7,'PRP':8,'VBG':9,'MD':10,'WDT':11,'DT':12}
            pos_obj_vocab = []
            for t in pos_sub_tokens_vocab:
                if t in pos_index_vocab.keys():# pos_vocab.stoi：这通常是一个字典，其中 stoi 可能是 “string to index” 的缩写，表示将字符串映射到索引的字典。
                    pos_obj_vocab.append(1)
                else:
                    pos_obj_vocab.append(0)
            # pos_obj_vocab = [label for t in obj['pos'] if t in pos_vocab.stoi.keys() for label, index in pos_index_vocab.items() if index == t]
                        
            # 创建分词后的最短路径矩阵
            tokenized_shortest_paths = np.zeros((len(bert_tokens), len(bert_tokens)), dtype='float32')
            # print("tokenized_shortest_paths shape1:",tokenized_shortest_paths.shape)
            # 通过 token2ori_map 映射原始路径的最短距离成分词矩阵最短路径
            for i in range(len(bert_tokens)):
                for j in range(len(bert_tokens)):
                    ori_i, ori_j = tok2ori_map[i], tok2ori_map[j]  # 映射到原始词汇
                    tokenized_shortest_paths[i][j] = D_min[ori_i][ori_j]  # 继承最短路径信息
            # print("tokenized_shortest_paths shape2:",tokenized_shortest_paths.shape)
            
            



            
            
            
            truncate_tok_len = len(bert_tokens) 
            tok_adj = np.zeros(
                (truncate_tok_len, truncate_tok_len), dtype='float32') # # 初始化调整后的邻接矩阵。
            # print("tok_adj shape1:",tok_adj.shape)
            
            # 根据原始邻接矩阵构建调整后的邻接矩阵
            for i in range(truncate_tok_len):
                for j in range(truncate_tok_len):
                    tok_adj[i][j] = ori_adj[tok2ori_map[i]][tok2ori_map[j]]
            # print("tok_adj shape2:",tok_adj.shape)

        
            
            # 生成动态prompt，例如："I feel the sentiment polarity of {term} is [MASK]"
            term_str = " ".join(term)  # 将aspect词列表转换为字符串（如"food"）
            # fake_polarity = ['positive','negative','neutral']
            # random.seed(opt.seed)
            # candidates = [p for p in fake_polarity if p != obj['label']]
            
            # random_fake_polarity = random.choice(fake_polarity)
            # prompt_str = f"I feel the sentiment polarity of {term_str} is {random_fake_polarity} , actually it is [MASK]"
            # prompt_str = f"Given the context , is {random_fake_polarity} the sentiment toward {term_str} ? Answer: [MASK]"
            # prompt_str = f"Given the context , which is the sentiment toward {term_str} ? Choose from [positive] [negative] [neutral] . Answer: [MASK]"
            # prompt_str = f"Given the context what is the sentiment toward {term_str} ? Answer: [MASK]"
            prompt_str = f"I feel the sentiment polarity of {term_str} is [MASK]"
            prompt_tokens = tokenizer.tokenize(prompt_str)  # 分词后的prompt

            # 调整输入序列构造：将原来的 term_tokens 替换为 prompt_tokens
            context_asp_ids = (
                [tokenizer.cls_token_id]
                + tokenizer.convert_tokens_to_ids(bert_tokens)  # 原始上下文
                + [tokenizer.sep_token_id]
                + tokenizer.convert_tokens_to_ids(prompt_tokens)  # 替换为prompt的分词
                + [tokenizer.sep_token_id]
            )
            
            # print("context_asp_ids:",len(context_asp_ids))

            context_asp_len = len(context_asp_ids)
            paddings = [0] * (tokenizer.max_seq_len - context_asp_len)# 计算填充的长度
            
            # print("tokenizer.max_seq_len:",tokenizer.max_seq_len)
            # print("paddings:",len(paddings))

            # 调整 segment IDs：prompt部分标记为1
            context_asp_seg_ids = (
                [0] * (1 + len(bert_tokens) + 1)
                + [1] * (len(prompt_tokens) + 1)  # prompt部分用segment_id=1
                + paddings
            )
            # print("context_asp_seg_ids:",len(context_asp_seg_ids))

            # 定位[MASK]的位置（用于后续分类）
            mask_position = len(bert_tokens) + 2 + prompt_tokens.index("[MASK]")
            # print("mask_position:",mask_position)

            # 调整 aspect_mask：关注[MASK]在完整输入序列中的全局位置。
            # 旧版本保留如下，方便后续回看。
            aspect_mask = [0] * (len(bert_tokens) + 2)  # CLS + 上下文 + SEP
            aspect_mask += [1 if i == mask_position else 0 for i in range(len(prompt_tokens) + 1)]
            aspect_mask += [0] * (opt.max_length - len(aspect_mask))
            # aspect_mask = [0] * opt.max_length
            # if 0 <= mask_position < opt.max_length:
            #     aspect_mask[mask_position] = 1
            # else:
            #     raise ValueError(
            #         f"mask_position out of range: {mask_position}, max_length={opt.max_length}"
            #     )
            
            # print("aspect_mask:",len(aspect_mask))
            
            # 构建方面 mask。aspect_mask2是不关注prompt的
            aspect_mask2 = [0] + [0] * asp_start + [1] * (asp_end - asp_start) 
            aspect_mask2 = aspect_mask2 + (opt.max_length - len(aspect_mask2)) * [0] 
            aspect_mask2 = np.asarray(aspect_mask2, dtype='int64')
            
            
            context_len = len(bert_tokens)
            # print("context_len:",context_len)
            
            
            # 构建源 mask。（src_mask 用于指示哪些 token 是有效的）
            # 修改后（扩展至 prompt）：
            # src_mask = [0] + [1] * (context_len + len(prompt_tokens) + 2) + [0] * (opt.max_length - context_len - len(prompt_tokens) - 3)   
            # src_mask不更改版本
            src_mask = [0] + [1] * context_len + [0] * (opt.max_length - context_len - 1)

            # print("src_mask:",len(src_mask))

            # 构建注意力 mask。
            context_asp_attention_mask = [1] * context_asp_len + paddings
            context_asp_ids += paddings
            # print("context_asp_attention_mask:",len(context_asp_attention_mask))
            # print("context_asp_ids:",len(context_asp_ids))
            # print("############################################################")
            
            # 将各种列表转换为 NumPy 数组。
            context_asp_ids = np.asarray(context_asp_ids, dtype='int64')
            context_asp_seg_ids = np.asarray(context_asp_seg_ids, dtype='int64')
            context_asp_attention_mask = np.asarray(context_asp_attention_mask, dtype='int64')
            src_mask = np.asarray(src_mask, dtype='int64')
            aspect_mask = np.asarray(aspect_mask, dtype='int64')
            # pad adj 
            # 初始化上下文-方面邻接矩阵，并填充调整后的邻接矩阵。
            context_asp_adj_matrix = np.zeros(
                (tokenizer.max_seq_len, tokenizer.max_seq_len)).astype('float32')
            # 第一行和第一列通常用于表示特殊的 token（如 CLS token），或者可能会留空。
            # 从第二行和第二列开始，表示实际的上下文 token。        
            context_asp_adj_matrix[1:context_len + 1, 1:context_len + 1] = tok_adj
            # print("tok_adj shape:",tok_adj.shape)
            # print("context_asp_adj_matrix shape:",context_asp_adj_matrix.shape)


            # 效仿上一行
            D_min_ = np.zeros(
                (tokenizer.max_seq_len, tokenizer.max_seq_len)).astype('float32')
            # print("D_min:",D_min.shape)
            # print("D_min_ shape:",D_min_.shape)
            D_min_[1:context_len + 1, 1:context_len + 1] = tokenized_shortest_paths


            M_pos = [0] * (tokenizer.max_seq_len)
            M_pos = np.array(M_pos)
            M_pos[1:context_len + 1] = pos_obj_vocab
            
            
            
            # 逻辑或融合两种aspect_mask
            com_aspect_mask = aspect_mask | aspect_mask2
            
            


            # 将所有处理后的信息存储到字典中，并添加到数据列表中
            data = {
                'text_bert_indices': context_asp_ids,
                'bert_segments_ids': context_asp_seg_ids,
                'attention_mask': context_asp_attention_mask,
                'asp_start': asp_start, # 分词后的起始位置
                'asp_end': asp_end, # 分词后的结束位置
                'adj_matrix': context_asp_adj_matrix,
                'D_min': D_min_,
                'M_pos': M_pos,
                'src_mask': src_mask,
                'aspect_mask': aspect_mask,
                'aspect_mask2': aspect_mask2,
                'com_aspect_mask': com_aspect_mask,
                'mask_position': mask_position,
                'polarity': polarity,
                'sentence_id': len(self.data),
                'text_list': text,
            }
            self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
