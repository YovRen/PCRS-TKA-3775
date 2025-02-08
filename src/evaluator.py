import re
import math
import torch
import json
from nltk import ngrams
from nltk.translate.bleu_score import sentence_bleu
slot_pattern = re.compile(r'<movie>')


class RecEvaluator:
    def __init__(self, k_list=None, device=torch.device('cpu')):
        if k_list is None:
            k_list = [1, 5, 10, 20, 50]
        self.k_list = k_list
        self.device = device

        self.metric = {}
        self.reset_metric()

    def evaluate(self, logits, labels):
        for logit, label in zip(logits, labels):
            for k in self.k_list:
                self.metric[f'recall@{k}'] += self.compute_recall(logit, label, k)
                self.metric[f'mrr@{k}'] += self.compute_mrr(logit, label, k)
                self.metric[f'ndcg@{k}'] += self.compute_ndcg(logit, label, k)
            self.metric['count'] += 1

    def compute_recall(self, rank, label, k):
        return int(label in rank[:k])

    def compute_mrr(self, rank, label, k):
        if label in rank[:k]:
            label_rank = rank.index(label)
            return 1 / (label_rank + 1)
        return 0

    def compute_ndcg(self, rank, label, k):
        if label in rank[:k]:
            label_rank = rank.index(label)
            return 1 / math.log2(label_rank + 2)
        return 0

    def reset_metric(self):
        for metric in ['recall', 'ndcg', 'mrr']:
            for k in self.k_list:
                self.metric[f'{metric}@{k}'] = 0
        self.metric['count'] = 0

    def report(self):
        report = {}
        for k, v in self.metric.items():
            report[k] = torch.tensor(v, device=self.device)[None]
        return report


class ConvEvaluator:
    def __init__(self, tokenizer, output_dir=None):
        self.tokenizer = tokenizer
        self.reset_metric()
        if output_dir:
            self.log_file = open(output_dir+'/gen_results.txt', 'w', buffering=1, encoding="utf-8")

    def evaluate(self, preds, labels, log=False):
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=False)
        decoded_preds = [decoded_pred.replace('<pad>', '').replace('<|endoftext|>', '') for decoded_pred in decoded_preds]
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=False)
        decoded_labels = [decoded_label.replace('<pad>', '').replace('<|endoftext|>', '') for decoded_label in decoded_labels]
        decoded_labels = [label.strip() for label in decoded_labels]

        if log and hasattr(self, 'log_file') and self.log_file is not None:
            for pred, label in zip(decoded_preds, decoded_labels):
                self.log_file.write(json.dumps({'pred': pred, 'label': label}, ensure_ascii=False) + '\n')

        self.collect_ngram(decoded_preds)
        self.compute_item_ratio(decoded_preds)
        self.compute_bleu(decoded_preds, decoded_labels)
        self.sent_cnt += len([pred for pred in decoded_preds if len(pred) > 0])

    def collect_ngram(self, strs):
        for str in strs:
            str = str.split()  # "I am preer"
            for k in range(1, 5):
                dist_k = f'dist@{k}'
                for token in ngrams(str, k):
                    self.metric[dist_k].add(token)

    def compute_bleu(self, preds, labels):
        for pred, label in zip(preds, labels):
            pred, label = pred.split(), [label.split()]
            for k in range(4):
                weights = [0] * 4
                weights[k] = 1
                self.metric[f'bleu@{k + 1}'] += sentence_bleu(label, pred, weights)

    def compute_item_ratio(self, strs):
        for str in strs:
            items = re.findall(slot_pattern, str)
            self.metric['item_ratio'] += len(items)

    def report(self):
        report = {}
        for k, v in self.metric.items():
            if self.sent_cnt == 0:
                report[k] = 0
            else:
                if 'dist' in k:
                    v = len(v)
                report[k] = v / self.sent_cnt
        report['sent_cnt'] = self.sent_cnt
        return report

    def reset_metric(self):
        self.metric = {'bleu@1': 0, 'bleu@2': 0, 'bleu@3': 0, 'bleu@4': 0, 'dist@1': set(), 'dist@2': set(), 'dist@3': set(), 'dist@4': set(), 'item_ratio': 0, }
        self.sent_cnt = 0
