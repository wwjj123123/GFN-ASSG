'''
Description: 
version: 
Author: chenhao
Date: 2021-06-09 14:17:37
'''
import os
import sys
import copy
import random
import logging
import argparse
import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics 
from time import strftime, localtime
from torch.utils.data import DataLoader
from transformers import BertModel, AdamW
from models.ian import IAN
from models.atae_lstm import ATAE_LSTM
from models.syngcn import SynGCNClassifier
from models.semgcn import SemGCNClassifier
from models.dualgcn import DualGCNClassifier
from models.dualgcn_bert import DualGCNBertClassifier
from models.no_aspDis_gcn import NoAspDisGCNClassifier
from models.no_posAtten_gcn import NoPosAttenGCNClassifier
from models.no_posAtten_gcn_bert import NoPosAttenGCNBertClassifier
from models.dapgcn import DAPGCNClassifier
from models.dapgcn_bert import DAPGCNBertClassifier

from models.triplegcn import TripleGCNClassifier

from data_utils import SentenceDataset, build_tokenizer, build_embedding_matrix, Tokenizer4BertGCN, ABSAGCNData
from prepare_vocab import VocabHelp

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class Instructor:
    ''' Model training and evaluation '''
    def __init__(self, opt):
        self.opt = opt
        if 'bert' in opt.model_name:
            tokenizer = Tokenizer4BertGCN(opt.max_length, opt.pretrained_bert_name)
            # from_pretrained() 是一个类方法，允许从 Hugging Face 模型库中下载并加载已经预训练好的模型权重和配置。该方法会自动从指定的模型名称或路径下载相应的权重，并且加载到模型中。
                # BertModel 返回的模型是一个基础的 BERT 模型，通常包含以下几个主要部分：
                # Embedding Layer：将输入的文本转换成 BERT 模型可以理解的嵌入表示。
                # Transformer Layers：多个 BERT 的 Transformer 层（编码器层），用于上下文信息的建模。
                # 输出：模型的最终输出是每个输入 token 的上下文感知表示。
            bert = BertModel.from_pretrained(opt.pretrained_bert_name)
            self.model = opt.model_class(bert, opt, tokenizer).to(opt.device)
            trainset = ABSAGCNData(opt.dataset_file['train'], tokenizer, opt=opt)
            testset = ABSAGCNData(opt.dataset_file['test'], tokenizer, opt=opt)
        else:    
            tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train'], opt.dataset_file['test']], 
                max_length=opt.max_length, 
                data_file='{}/{}_tokenizer.dat'.format(opt.vocab_dir, opt.dataset))
            embedding_matrix = build_embedding_matrix(
                vocab=tokenizer.vocab, 
                embed_dim=opt.embed_dim, 
                data_file='{}/{}d_{}_embedding_matrix.dat'.format(opt.vocab_dir, str(opt.embed_dim), opt.dataset))

            logger.info("Loading vocab...")
            print('Loading vocab...')
            token_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_tok.vocab')    # token
            post_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_post.vocab')    # position
            pos_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_pos.vocab')      # POS
            dep_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_dep.vocab')      # deprel
            pol_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_pol.vocab')      # polarity
            logger.info("token_vocab: {}, post_vocab: {}, pos_vocab: {}, dep_vocab: {}, pol_vocab: {}".format(len(token_vocab), len(post_vocab), len(pos_vocab), len(dep_vocab), len(pol_vocab)))
            print("token_vocab: {}, post_vocab: {}, pos_vocab: {}, dep_vocab: {}, pol_vocab: {}".format(len(token_vocab), len(post_vocab), len(pos_vocab), len(dep_vocab), len(pol_vocab)))
            
            # opt.tok_size = len(token_vocab)
            opt.post_size = len(post_vocab)
            opt.pos_size = len(pos_vocab)
            
            vocab_help = (post_vocab, pos_vocab, dep_vocab, pol_vocab)
            self.model = opt.model_class(embedding_matrix, opt).to(opt.device)
            trainset = SentenceDataset(opt.dataset_file['train'], tokenizer, opt=opt, vocab_help=vocab_help)
            testset = SentenceDataset(opt.dataset_file['test'], tokenizer, opt=opt, vocab_help=vocab_help)

        self.train_dataloader = DataLoader(dataset=trainset, batch_size=opt.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(dataset=testset, batch_size=opt.batch_size)

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(self.opt.device.index)))
            print('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(self.opt.device.index)))
        self._print_args()
    
    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += int(n_params)# 这部分在训练no_posAtten_gcn最好数据前没有改过（现在以改动）
            else:
                n_nontrainable_params += int(n_params)# 这部分在训练no_posAtten_gcn最好数据前没有改过（现在以改动）

        logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('training arguments:') # 打印训练参数配置
        print('training arguments:')
        
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))
    
    
    
    
    
    
    
    # 这部分在训练no_posAtten_gcn最好数据前没有改过（现在以改动）
    def _reset_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape) > 0:  # 检查是否有维度
                    if len(p.shape) > 1:
                        self.opt.initializer(p)
                    else:
                        stdv = 1. / (p.shape[0] ** 0.5) if p.shape[0] > 0 else 0  # 避免除以零
                        torch.nn.init.uniform_(p, a=-stdv, b=stdv)
                else:
                    print("Warning: Parameter with no dimensions found, skipping initialization.")


    # 给bert设置优化器
    def get_bert_optimizer(self, model):
        # Prepare optimizer and schedule (linear warmup and decay)
        # 包含 'bias' 和 'LayerNorm.weight' 的参数不会使用权重衰减（因为这些参数通常不包含有用的正则化信息）。
        no_decay = ['bias', 'LayerNorm.weight'] 
        # 指定了 BERT 模型中两个关键部分："bert.embeddings"（词嵌入层）和 "bert.encoder"（编码器层）。这些部分可能会使用不同的学习率设置。
        diff_part = ["bert.embeddings", "bert.encoder"]

        # 采用分层学习率策略，模型的不同部分将使用不同的学习率和权重衰减配置
        if self.opt.diff_lr: 
            logger.info("layered learning rate on")
            # 分四层策略
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if
                            not any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
                    "weight_decay": self.opt.weight_decay,
                    "lr": self.opt.bert_lr
                },
                {
                    "params": [p for n, p in model.named_parameters() if
                            any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
                    "weight_decay": 0.0,
                    "lr": self.opt.bert_lr
                },
                {
                    "params": [p for n, p in model.named_parameters() if
                            not any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
                    "weight_decay": self.opt.weight_decay,
                    "lr": self.opt.learning_rate
                },
                {
                    "params": [p for n, p in model.named_parameters() if
                            any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
                    "weight_decay": 0.0,
                    "lr": self.opt.learning_rate
                },
            ]
            optimizer = AdamW(optimizer_grouped_parameters, eps=self.opt.adam_epsilon)
        # 不使用分层学习率，而是为整个模型使用相同的学习率策略
        else:
            logger.info("bert learning rate on")
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.opt.weight_decay},
                {'params': [p for n, p in model.named_parameters() if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.opt.bert_lr, eps=self.opt.adam_epsilon)

        return optimizer

    # 计算类别权重
    def compute_class_weights(self, train_loader, num_classes=3):
        class_counts = [0] * num_classes
        polarity_dict = {'positive':0, 'negative':1, 'neutral':2}
        # 统计每个类别的出现频次
        for i_batch, sample_batched in enumerate(train_loader):
            classes = sample_batched['polarity'].to(self.opt.device)
            # 遍历当前批次中的每个标签
            for label in classes:
                class_counts[label.item()] += 1  # 更新类别的计数 # label.item() 不是用来取出键值的，它是用来从单一元素的 PyTorch 张量中提取标量值        
        print(f'positive_num:{class_counts[0]}, negative_num:{class_counts[1]}, neutral_num:{class_counts[2]}')
        
        # 计算每个类别的权重，频率较少的类别权重大
        total_count = sum(class_counts)
        class_weights = [total_count/(count+total_count)**2 for count in class_counts]
        class_weights_tensor = torch.tensor(class_weights).float()
        normalized_class_weights = class_weights_tensor/class_weights_tensor[0] 
        # 将权重转换为Tensor，并移到GPU上（如果需要）
        normalized_class_weights = torch.tensor(normalized_class_weights).cuda()  # 如果在GPU上训练，确保权重也在GPU上
        

        return normalized_class_weights

    
    def _train(self, optimizer, max_test_acc_overall=0):
        print("Starting training loop...")
        max_test_acc = 0
        max_f1 = 0
        global_step = 0
        model_path = ''

        # 初始化类别权重为可学习参数
        # classes_weight = self.compute_class_weights(self.train_dataloader, num_classes=3)
        # change_formula
        # classes_weight = torch.tensor([1.0, 1.2, 1.6]).cuda() # 
        classes_weight = torch.tensor([1.0, 1.2, 1.45]).cuda() # laptop
        # classes_weight = torch.tensor([1.0, 1.6, 1.6]).cuda() # rest14
        # classes_weight = torch.tensor([1.0, 1.0, 0.65]).cuda() # twitter



        # class_weights = torch.tensor([1.0, 2.0, 1.0]).cuda() # twitter15
        # class_weights = torch.tensor([1.0, 3.0, 1.0]).cuda() # twitter17
        class_weights = torch.tensor([1.0, 1.2, 1.0]).cuda() # MAMS




        
        # No_change_formula
        # classes_weight = torch.tensor([1.0, 1.15, 1.5]).cuda() # laptop
        # classes_weight = torch.tensor([1.0, 1.6, 1.6]).cuda() # rest14
        # classes_weight = torch.tensor([1.0, 1.0, 0.9]).cuda() # twitter
        print(f"classes_weight:{class_weights}")  
        criterion = nn.CrossEntropyLoss(weight=classes_weight)
        print(f"alpha: {self.opt.alpha}, beta: {self.opt.beta} ")
        
        for epoch in range(self.opt.num_epoch):
            logger.info('>' * 60)
            print('>' * 60) 
            logger.info('epoch: {}'.format(epoch))
            print('epoch: {}'.format(epoch))
            n_correct, n_total = 0, 0
            for i_batch, sample_batched in enumerate(self.train_dataloader):
                global_step += 1
                # switch model to training mode, clear gradient accumulators
                self.model.train()
                optimizer.zero_grad()
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs, penal = self.model(inputs)
                targets = sample_batched['polarity'].to(self.opt.device)
                if self.opt.losstype is not None:
                    loss = criterion(outputs, targets) + penal
                else:
                    loss = criterion(outputs, targets)

                #     # 辅助损失：Recall/F1目标
                # pred = outputs.argmax(dim=-1)
                # recall = metrics.recall_score(targets.cpu(), pred.cpu(), average='macro')
                # f1 = metrics.f1_score(targets.cpu(), pred.cpu(), average='macro')
                # loss_aux = 1 - recall  # 最大化Recall（反向为损失）
                
                # # 总损失
                # alpha = 1.0
                # beta = 0.1
                # loss = alpha * loss + beta * loss_aux

                loss.backward()
                optimizer.step()
                
                if global_step % self.opt.log_step == 0:
                    print(f"Global Step: {global_step}")
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    train_acc = n_correct / n_total
                    print(f"Train Acc: {train_acc}")
                    test_acc, f1 = self._evaluate()
                    print(f"Test Acc: {test_acc}, F1: {f1}")
                    if test_acc > max_test_acc or (abs(test_acc - max_test_acc) < 0.0000000001 and f1>max_f1):
                        max_test_acc = test_acc
                        if test_acc > max_test_acc_overall:
                            if not os.path.exists('./DualGCN/state_dict_correct_prompt_newDmin_softmax'):
                                os.mkdir('./DualGCN/state_dict_correct_prompt_newDmin_softmax')
                            model_path = './DualGCN/state_dict_correct_prompt_newDmin_softmax/{}_{}_acc_{:.4f}_f1_{:.4f}'.format(self.opt.model_name, self.opt.dataset, test_acc, f1)
                            self.best_model = copy.deepcopy(self.model)
                            logger.info('>> saved: {}'.format(model_path))
                            print('>> saved: {}'.format(model_path))
                    # if f1 >0.7563 and f1<0.758:
                    #     if not os.path.exists('./DualGCN/state_dict_correct'):
                    #         os.mkdir('./DualGCN/state_dict_correct')
                    #     model_path = './DualGCN/state_dict_correct/{}_{}_acc_{:.4f}_f1_{:.4f}'.format(self.opt.model_name, self.opt.dataset, test_acc, f1)
                    #     self.best_model = copy.deepcopy(self.model)
                    #     logger.info('>> saved: {}'.format(model_path))
                    #     print('>> saved: {}'.format(model_path))
                    # 查看此时的alpha
                    if self.opt.model_name == 'no_aspDis_gcn' or self.opt.model_name == 'dapgcn':
                        alpha_value = self.model.gcn_model.gcn.alpha.item()  # 获取 alpha 的值
                        print('alpha_value: {}'.format(alpha_value))     
                        
                    if f1 > max_f1:
                        max_f1 = f1
                    logger.info('loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, f1: {:.4f}'.format(loss.item(), train_acc, test_acc, f1))
                    print('loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, f1: {:.4f}'.format(loss.item(), train_acc, test_acc, f1))
        logger.info('Training complete.')  # 添加日志
        print('Training complete.')
        return max_test_acc, max_f1, model_path
    
    def _evaluate(self, show_results=False):
        # switch model to evaluation mode
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        targets_all, outputs_all = None, None
        with torch.no_grad():
            for batch, sample_batched in enumerate(self.test_dataloader):
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                targets = sample_batched['polarity'].to(self.opt.device)
                outputs, penal = self.model(inputs)
                n_test_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_test_total += len(outputs)
                targets_all = torch.cat((targets_all, targets), dim=0) if targets_all is not None else targets
                outputs_all = torch.cat((outputs_all, outputs), dim=0) if outputs_all is not None else outputs
        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(targets_all.cpu(), torch.argmax(outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')

        labels = targets_all.data.cpu()
        predic = torch.argmax(outputs_all, -1).cpu()
        print('Evaluation complete.')
        if show_results:
            report = metrics.classification_report(labels, predic, digits=4)
            confusion = metrics.confusion_matrix(labels, predic)
            return report, confusion, test_acc, f1

        return test_acc, f1

    def _test(self):
        self.model = self.best_model
        self.model.eval()
        test_report, test_confusion, acc, f1 = self._evaluate(show_results=True)
        logger.info("Precision, Recall and F1-Score...") # 记录评估结果
        print("Precision, Recall and F1-Score...")
        logger.info(test_report)
        print(test_report)
        logger.info("Confusion Matrix...")
        print("Confusion Matrix...")
        logger.info(test_confusion)
        print(test_confusion)
        
    
    def run(self):
        print("Starting run method...")
        # 类权重（根据训练集中各类的不平衡调整）
        # class_weights = torch.tensor([1.0, 1.425, 1.25]).cuda()  # 根据实际情况调整
        # criterion = nn.CrossEntropyLoss(weight=class_weights)
        # criterion = nn.CrossEntropyLoss()
        if 'bert' not in self.opt.model_name:
            _params = filter(lambda p: p.requires_grad, self.model.parameters())
            optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
        else:
            optimizer = self.get_bert_optimizer(self.model)
        max_test_acc_overall = 0
        max_f1_overall = 0
        if 'bert' not in self.opt.model_name:
            self._reset_params()
        print("Starting training...")
        max_test_acc, max_f1, model_path = self._train(optimizer, max_test_acc_overall)
        print("Training completed.")
        logger.info('max_test_acc: {0}, max_f1: {1}'.format(max_test_acc, max_f1))
        print('max_test_acc: {0}, max_f1: {1}'.format(max_test_acc, max_f1))
        max_test_acc_overall = max(max_test_acc, max_test_acc_overall)
        max_f1_overall = max(max_f1, max_f1_overall)
        torch.save(self.best_model.state_dict(), model_path)
        logger.info('>> saved: {}'.format(model_path))
        print('>> saved: {}'.format(model_path))
        logger.info('#' * 60)
        logger.info('max_test_acc_overall:{}'.format(max_test_acc_overall))
        logger.info('max_f1_overall:{}'.format(max_f1_overall))
        print('#' * 60)
        print('max_test_acc_overall:{}'.format(max_test_acc_overall))
        print('max_f1_overall:{}'.format(max_f1_overall))
        self._test()


def main():
    model_classes = {
        'atae_lstm': ATAE_LSTM,
        'ian': IAN,
        'syngcn': SynGCNClassifier,
        'semgcn': SemGCNClassifier,
        'dualgcn': DualGCNClassifier,
        'dualgcn_bert': DualGCNBertClassifier,
        'no_posAtten_gcn': NoPosAttenGCNClassifier,
        'no_aspDis_gcn': NoAspDisGCNClassifier,
        'dapgcn': DAPGCNClassifier,
        'no_posAtten_gcn_bert': NoPosAttenGCNBertClassifier,
        'dapgcn_bert': DAPGCNBertClassifier,
        'triplegcn': TripleGCNClassifier
        
    }
    
    dataset_files = {
        'restaurant': {
            'train': './DualGCN/dataset/Restaurants_corenlp/train.json',
            'test': './DualGCN/dataset/Restaurants_corenlp/test.json',
        },
        'laptop': {
            'train': './DualGCN/dataset/Laptops_corenlp/train.json',
            'test': './DualGCN/dataset/Laptops_corenlp/test.json'
        },
        'twitter': {
            'train': './DualGCN/dataset/Tweets_corenlp/train.json',
            'test': './DualGCN/dataset/Tweets_corenlp/test.json',
        },
        'twitter2015': {
            'train': './DualGCN/dataset/Twitter2015_corenlp/train.json',
            'test': './DualGCN/dataset/Twitter2015_corenlp/test.json',
        },
        'twitter2017': {
            'train': './DualGCN/dataset/Twitter2017_corenlp/train.json',
            'test': './DualGCN/dataset/Twitter2017_corenlp/test.json',
        },
        'MAMS': {
            'train': './DualGCN/dataset/MAMS/train.json',
            'test': './DualGCN/dataset/MAMS/test.json',
        }
    }
    
    input_colses = {
        'atae_lstm': ['text', 'aspect'],
        'ian': ['text', 'aspect'],
        'syngcn': ['text', 'aspect', 'pos', 'head', 'deprel', 'post', 'mask', 'length', 'adj'],
        'semgcn': ['text', 'aspect', 'pos', 'head', 'deprel', 'post', 'mask', 'length'],
        'dualgcn': ['text', 'aspect', 'pos', 'head', 'deprel', 'post', 'mask', 'length', 'adj', 'asp_index', 'polarity'],
        'dualgcn_bert': ['text_bert_indices', 'bert_segments_ids', 'attention_mask', 'asp_start', 'asp_end', 'adj_matrix', 'src_mask', 'aspect_mask'],
        'no_posAtten_gcn_bert': ['text_bert_indices', 'bert_segments_ids', 'attention_mask', 'asp_start', 'asp_end', 'adj_matrix', 'D_min', 'src_mask', 'aspect_mask'],
        'dapgcn_bert': ['text_bert_indices', 'bert_segments_ids', 'attention_mask', 'asp_start', 'asp_end', 'adj_matrix', 'D_min','M_pos', 'src_mask', 'aspect_mask','aspect_mask2','com_aspect_mask', 'mask_position', 'sentence_id'], # 

        'no_posAtten_gcn': ['text', 'aspect', 'pos', 'head', 'deprel', 'post', 'mask', 'length', 'adj', 'pos_obj_vocab', 'ped_post', 'asp_index', 'polarity'],
        'triplegcn': ['text', 'aspect', 'pos', 'head', 'deprel', 'post', 'mask', 'length', 'adj', 'pos_obj_vocab', 'ped_post', 'asp_index', 'polarity'],
        'no_aspDis_gcn': ['text', 'aspect', 'pos', 'head', 'deprel', 'post', 'mask', 'length', 'adj', 'pos_obj_vocab', 'ped_post', 'asp_index','polarity'],
        'dapgcn': ['text', 'aspect', 'pos', 'head', 'deprel', 'post', 'mask', 'length', 'adj', 'pos_obj_vocab', 'ped_post', 'asp_index','polarity']

    }
    
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    
    optimizers = {
        'adadelta': torch.optim.Adadelta,
        'adagrad': torch.optim.Adagrad, 
        'adam': torch.optim.Adam,
        'adamax': torch.optim.Adamax, 
        'asgd': torch.optim.ASGD,
        'rmsprop': torch.optim.RMSprop,
        'sgd': torch.optim.SGD,
    }
    
    # Hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='dualgcn', type=str, help=', '.join(model_classes.keys()))
    parser.add_argument('--dataset', default='laptop', type=str, help=', '.join(dataset_files.keys()))
   
    parser.add_argument('--pretrained_model_path', default=None, type=str, help="Path to the pretrained model (.pt file)")
    
    parser.add_argument('--optimizer', default='adam', type=str, help=', '.join(optimizers.keys()))
    parser.add_argument('--initializer', default='xavier_uniform_', type=str, help=', '.join(initializers.keys()))
    parser.add_argument('--learning_rate', default=0.002, type=float)
    parser.add_argument('--l2reg', default=1e-4, type=float)
    parser.add_argument('--num_epoch', default=20, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--post_dim', type=int, default=30, help='Position embedding dimension.')
    parser.add_argument('--pos_dim', type=int, default=30, help='Pos embedding dimension.')
    parser.add_argument('--hidden_dim', type=int, default=50, help='GCN mem dim.')
    parser.add_argument('--num_layers', type=int, default=2, help='Num of GCN layers.')
    parser.add_argument('--polarities_dim', default=3, type=int, help='3')

    parser.add_argument('--input_dropout', type=float, default=0.7, help='Input dropout rate.')
    parser.add_argument('--gcn_dropout', type=float, default=0.1, help='GCN layer dropout rate.')
    parser.add_argument('--lower', default=True, help='Lowercase all words.')
    parser.add_argument('--direct', default=False, help='directed graph or undirected graph')
    parser.add_argument('--loop', default=True)

    parser.add_argument('--bidirect', default=True, help='Do use bi-RNN layer.')
    parser.add_argument('--rnn_hidden', type=int, default=50, help='RNN hidden state size.')
    parser.add_argument('--rnn_layers', type=int, default=1, help='Number of RNN layers.')
    parser.add_argument('--rnn_dropout', type=float, default=0.1, help='RNN dropout rate.')
    
    parser.add_argument('--attention_heads', default=1, type=int, help='number of multi-attention heads')
    parser.add_argument('--max_length', default=85, type=int)
    parser.add_argument('--device', default=None, type=str, help='cpu, cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument('--vocab_dir', type=str, default='./DualGCN/dataset/Laptops_corenlp')
    parser.add_argument('--pad_id', default=0, type=int)
    parser.add_argument('--parseadj', default=False, action='store_true', help='dependency probability')
    parser.add_argument('--parsehead', default=False, action='store_true', help='dependency tree')
    parser.add_argument('--cuda', default='0', type=str)
    parser.add_argument('--losstype', default=None, type=str, help="['doubleloss', 'orthogonalloss', 'differentiatedloss']")
    parser.add_argument('--alpha', default=0.25, type=float)
    parser.add_argument('--beta', default=0.25, type=float)
    
    # * bert
    parser.add_argument('--pretrained_bert_name', default='/home/liugaofei/wyj/DualGCN-ABSA-main3/LAL-Parser/src_joint/bert-base-uncased', type=str)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--bert_dim', type=int, default=768)
    parser.add_argument('--bert_dropout', type=float, default=0.3, help='BERT dropout rate.')
    parser.add_argument('--diff_lr', default=False, action='store_true')
    parser.add_argument('--bert_lr', default=2e-5, type=float)
    parser.add_argument('--adj_alpha', default=0.8, type=float, help='the weight of distance')
    parser.add_argument('--adj_beta', default=0.4, type=float, help='the threshold that whether link aspect with words directly')
    parser.add_argument('--adj_gama', default=1.2, type=float, help='the weight of kl divergence loss')
    parser.add_argument('--fusion', default=True, type=bool,
                    help='fuse distance based weighted matrices belonging to different aspects')
    opt = parser.parse_args()
    	
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]

    # print("choice cuda:{}".format(opt.cuda))
    # os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if opt.device is None else torch.device(opt.device)
    
    # set random seed
    setup_seed(opt.seed)

    if not os.path.exists('./DualGCN/log'):
        os.makedirs('./DualGCN/log', mode=0o777)
    log_file = '{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%Y-%m-%d_%H:%M:%S", localtime()))
    logger.addHandler(logging.FileHandler("%s/%s" % ('./DualGCN/log', log_file)))

    ins = Instructor(opt)
    ins.run()

if __name__ == '__main__':
    main()
