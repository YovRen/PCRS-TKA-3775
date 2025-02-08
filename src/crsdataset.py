from collections import defaultdict
from copy import deepcopy

from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import json
import torch
from transformers import AutoTokenizer


class CRSDataset(Dataset):
    def __init__(self, data_path, split, task, entity_pad_idx=None, add_tree=False, ary2=None, ary1=None):
        self.task = task
        self.encoder_tokenizer = AutoTokenizer.from_pretrained('../model/roberta-base')
        self.encoder_tokenizer.add_special_tokens({'additional_special_tokens': ['<movie>'], })
        self.decoder_tokenizer = AutoTokenizer.from_pretrained('../model/dialogpt-small')
        self.decoder_tokenizer.add_special_tokens({'pad_token': "<pad>", 'additional_special_tokens': ['<movie>'], })
        self.max_p_length = 200
        self.max_r_length = 183
        self.max_e_length = 32
        self.max_t_length = 100
        self.entity_pad_idx = entity_pad_idx
        self.processed_cases = []
        f = open(data_path + split + '_data_processed.jsonl', encoding='utf-8')
        for i, lines in enumerate(tqdm(f)):
            lines = json.loads(lines.strip())
            if self.task == 'rec' or self.task == 'rec_pre':
                if self.task == 'rec' and ('System:' not in lines['response_text'] or len(lines['context_texts']) == 0):
                    continue
                data = dict()
                if add_tree:
                    tree_encoder_text = lines[f'context_entities_{ary1}_{ary2}_tree'] + self.encoder_tokenizer.sep_token
                    tree_encoder_ids = self.encoder_tokenizer.convert_tokens_to_ids(self.encoder_tokenizer.tokenize(tree_encoder_text))
                    data['tree_input_ids'] = [self.encoder_tokenizer.cls_token_id] + tree_encoder_ids[-(self.max_t_length - 1):]
                encoder_response_text = lines['response_text'] + self.encoder_tokenizer.sep_token
                decoder_response_text = lines['response_text'] + self.decoder_tokenizer.eos_token
                encoder_response_ids = self.encoder_tokenizer.convert_tokens_to_ids(self.encoder_tokenizer.tokenize(encoder_response_text))
                decoder_response_ids = self.decoder_tokenizer.convert_tokens_to_ids(self.decoder_tokenizer.tokenize(decoder_response_text))
                encoder_context_text = self.encoder_tokenizer.sep_token.join(lines['context_texts']) + self.encoder_tokenizer.sep_token if len(lines['context_texts']) > 0 else ""
                decoder_context_text = self.decoder_tokenizer.eos_token.join(lines['context_texts']) + self.decoder_tokenizer.eos_token if len(lines['context_texts']) > 0 else ""
                encoder_context_ids = self.encoder_tokenizer.convert_tokens_to_ids(self.encoder_tokenizer.tokenize(encoder_context_text))
                decoder_context_ids = self.decoder_tokenizer.convert_tokens_to_ids(self.decoder_tokenizer.tokenize(decoder_context_text))
                if self.task == 'rec_pre':
                    data['encoder_input_ids'] = [self.encoder_tokenizer.cls_token_id] + (encoder_context_ids + encoder_response_ids)[-(self.max_p_length - 1):]
                    data['decoder_input_ids'] = (decoder_context_ids + decoder_response_ids)[-self.max_p_length:]
                else:
                    data['encoder_input_ids'] = [self.encoder_tokenizer.cls_token_id] + encoder_context_ids[-(self.max_p_length - 1):]
                    data['decoder_input_ids'] = decoder_context_ids[-self.max_p_length:]
                data['context_entities'] = lines['context_entities'][-self.max_e_length:]
                response_entities = lines['response_entities'] if self.task == 'rec_pre' else lines['response_movies']
                for response_entity in response_entities:
                    data['response_entity'] = response_entity
                    self.processed_cases.append(deepcopy(data))
            elif self.task == 'conv' or self.task == 'conv_pre':
                if len(lines['context_texts']) == 0 or (self.task == 'conv' and 'System:' not in lines['response_text']):
                    continue
                data = dict()
                if add_tree:
                    tree_encoder_text = (lines[f'context_response_entities_{ary1}_{ary2}_tree'] if self.task == 'conv_pre' else lines[f'context_entities_{ary1}_{ary2}_tree']) + self.encoder_tokenizer.sep_token
                    tree_encoder_ids = self.encoder_tokenizer.convert_tokens_to_ids(self.encoder_tokenizer.tokenize(tree_encoder_text))
                    data['tree_input_ids'] = [self.encoder_tokenizer.cls_token_id] + tree_encoder_ids[-(self.max_t_length - 1):]
                encoder_response_text_masked = lines['response_text_masked']
                decoder_response_text_masked = lines['response_text_masked']
                encoder_response_ids = self.encoder_tokenizer.convert_tokens_to_ids(self.encoder_tokenizer.tokenize(encoder_response_text_masked))
                decoder_response_ids = self.decoder_tokenizer.convert_tokens_to_ids(self.decoder_tokenizer.tokenize(decoder_response_text_masked))
                encoder_context_text = self.encoder_tokenizer.sep_token.join(lines['context_texts']) + self.encoder_tokenizer.sep_token if len(lines['context_texts']) > 0 else ""
                decoder_context_text = self.decoder_tokenizer.eos_token.join(lines['context_texts']) + self.decoder_tokenizer.eos_token if len(lines['context_texts']) > 0 else ""
                encoder_context_ids = self.encoder_tokenizer.convert_tokens_to_ids(self.encoder_tokenizer.tokenize(encoder_context_text))
                decoder_context_ids = self.decoder_tokenizer.convert_tokens_to_ids(self.decoder_tokenizer.tokenize(decoder_context_text))
                decoder_prompt_ids = self.decoder_tokenizer.convert_tokens_to_ids(self.decoder_tokenizer.tokenize('System:' if lines['response_text'].startswith('System:') else 'User:'))
                if self.task == 'conv_pre':
                    data['encoder_input_ids'] = [self.encoder_tokenizer.cls_token_id] + (encoder_context_ids + encoder_response_ids[:self.max_r_length - 1] + [self.encoder_tokenizer.sep_token_id])[-(self.max_p_length + self.max_r_length - 1):]
                else:
                    data['encoder_input_ids'] = [self.encoder_tokenizer.cls_token_id] + encoder_context_ids[-(self.max_p_length - 1):]
                data['decoder_input_ids'] = (decoder_context_ids + decoder_response_ids[:self.max_r_length - 1] + [self.decoder_tokenizer.eos_token_id])[-(self.max_p_length + self.max_r_length):]
                data['decoder_gen_input_ids'] = decoder_context_ids[-(self.max_p_length - len(decoder_prompt_ids)):] + decoder_prompt_ids
                data['decoder_gen_label_ids'] = decoder_response_ids[:self.max_r_length - 1] + [self.decoder_tokenizer.eos_token_id]
                data['decoder_gen_input_ids_len'] = len(data['decoder_gen_input_ids']) - len(decoder_prompt_ids)
                data['context_entities'] = (lines['context_response_entities'] if self.task == 'conv_pre' else lines['context_entities'])[-self.max_e_length:]
                data['response_entity'] = lines['response_entities'][-1] if len(lines['response_entities']) > 0 else self.entity_pad_idx
                self.processed_cases.append(deepcopy(data))
            else:
                assert False

    def __getitem__(self, index):
        return self.processed_cases[index]

    def __len__(self):
        return len(self.processed_cases)


def padded_tensor(items, pad_idx=0, pad_tail=True, max_len=None, device=torch.device('cpu')):
    max_len = max(max([len(item) for item in items]), max_len if max_len is not None else 0)
    output = torch.full((len(items), max_len), pad_idx, dtype=torch.long, device=device)
    for i, item in enumerate(items):
        if len(item) == 0:
            continue
        item = torch.tensor(item, dtype=torch.long, device=device) if not isinstance(item, torch.Tensor) else item
        if pad_tail:
            output[i, :len(item)] = item
        else:
            output[i, -len(item):] = item
    return output


class CRSDataCollator:
    def __init__(self, task, device, entity_pad_idx=None, enriched_entity_pad_idx=None, add_tree=False):
        self.task = task
        self.device = device
        self.entity_pad_idx = entity_pad_idx
        self.enriched_entity_pad_idx = enriched_entity_pad_idx
        self.encoder_tokenizer = AutoTokenizer.from_pretrained('../model/roberta-base')
        self.encoder_tokenizer.add_special_tokens({'additional_special_tokens': ['<movie>'], })
        self.decoder_tokenizer = AutoTokenizer.from_pretrained('../model/dialogpt-small')
        self.decoder_tokenizer.add_special_tokens({'pad_token': "<pad>", 'additional_special_tokens': ['<movie>'], })
        self.add_tree = add_tree

    def __call__(self, datas):
        tree_input_ids_batch = []
        encoder_input_ids_batch = []
        decoder_input_ids_batch = []
        decoder_gen_input_ids_batch = []
        decoder_gen_input_ids_len_batch = []
        decoder_gen_label_ids_batch = []
        context_entities_batch = []
        response_entity_batch = []
        data_batch = defaultdict(lambda: defaultdict())
        for data in datas:
            context_entities_batch.append(data['context_entities'])
            response_entity_batch.append(data['response_entity'])
            if self.add_tree:
                tree_input_ids_batch.append(data['tree_input_ids'])
            encoder_input_ids_batch.append(data['encoder_input_ids'])
            decoder_input_ids_batch.append(data['decoder_input_ids'])
            if self.task == 'conv' or self.task == 'conv_pre':
                decoder_gen_input_ids_batch.append(data['decoder_gen_input_ids'])
                decoder_gen_label_ids_batch.append(data['decoder_gen_label_ids'])
                decoder_gen_input_ids_len_batch.append(data['decoder_gen_input_ids_len'])
        if self.add_tree:
            data_batch['tree']['input_ids'] = tree_input_ids_batch
            data_batch['tree'] = self.encoder_tokenizer.pad(data_batch['tree'], padding=True, return_tensors='pt').to(self.device)
        data_batch['encoder']['input_ids'] = encoder_input_ids_batch
        data_batch['encoder'] = self.encoder_tokenizer.pad(data_batch['encoder'], padding=True, return_tensors='pt').to(self.device)
        data_batch['decoder']['input_ids'] = decoder_input_ids_batch
        data_batch['decoder'] = self.decoder_tokenizer.pad(data_batch['decoder'], padding=True, return_tensors='pt').to(self.device)
        data_batch['prompt']['token_input_ids'] = data_batch['encoder']['input_ids']
        if self.add_tree:
            data_batch['prompt']['tree_input_ids'] = data_batch['tree']['input_ids']
        data_batch['prompt']['entity_ids'] = padded_tensor(context_entities_batch, pad_idx=self.entity_pad_idx, pad_tail=True, device=self.device)
        data_batch['prompt']['rec_labels'] = torch.tensor(response_entity_batch).to(self.device)
        if self.task == 'rec' or self.task == 'rec_pre':
            data_batch['decoder']['rec_labels'] = torch.tensor(response_entity_batch).to(self.device)
            for k, v in data_batch['decoder'].items():
                if not isinstance(v, torch.Tensor):
                    data_batch['decoder'][k] = torch.as_tensor(v, device=self.device)
            return data_batch
        elif self.task == 'conv' or self.task == 'conv_pre':
            data_batch['decoder']['conv_labels'] = [[token_id if token_id != self.decoder_tokenizer.pad_token_id else -100 for token_id in resp] for resp in data_batch['decoder']['input_ids']]
            for k, v in data_batch['decoder'].items():
                if not isinstance(v, torch.Tensor):
                    data_batch['decoder'][k] = torch.as_tensor(v, device=self.device)
            self.decoder_tokenizer.padding_side = 'left'
            data_batch['gen_decoder']['input_ids'] = decoder_gen_input_ids_batch
            data_batch['gen_decoder'] = self.decoder_tokenizer.pad(data_batch['gen_decoder'], padding=True)
            self.decoder_tokenizer.padding_side = 'right'
            for k, v in data_batch['gen_decoder'].items():
                if not isinstance(v, torch.Tensor) and k != 'conv_labels':
                    data_batch['gen_decoder'][k] = torch.as_tensor(v, device=self.device)
            data_batch['gen_decoder_label_ids'] = decoder_gen_label_ids_batch
            data_batch['gen_decoder_input_ids_len'] = decoder_gen_input_ids_len_batch
            return data_batch
        else:
            assert False


class KGInfo:

    def __init__(self, data_path, device):
        self.device = device
        self.data_path = data_path

    def get_kg_info(self):
        self.dbpedia_subkg = json.load(open(self.data_path + 'dbpedia_subkg.json', 'rb'))
        self.entity_url2entity_idx = json.load(open(self.data_path + 'entity_url2entity_idx.json', encoding='utf-8'))
        self.relation_url2relation_idx = json.load(open(self.data_path + 'relation_url2relation_idx.json', encoding='utf-8'))
        self.movie_idxs = json.load(open(self.data_path + 'movie_idxs.json', encoding='utf-8'))
        edge_list = set()  # [(entity, entity, relation)]
        for entity_idx in self.entity_url2entity_idx.values():
            if str(entity_idx) not in self.dbpedia_subkg:
                continue
            for relation_and_tail in self.dbpedia_subkg[str(entity_idx)]:
                edge_list.add((entity_idx, relation_and_tail[1], relation_and_tail[0]))
                edge_list.add((relation_and_tail[1], entity_idx, relation_and_tail[0]))
        edge_list = list(edge_list)
        edge = torch.as_tensor(edge_list, dtype=torch.long, device=self.device)
        self.edge_index = edge[:, :2].t()
        self.edge_type = edge[:, 2]
        self.num_relations = len(self.relation_url2relation_idx)
        self.num_entities = len(self.entity_url2entity_idx) + 1
        self.entity_pad_idx = len(self.entity_url2entity_idx)

        kg_info = {'edge_index': self.edge_index,
                   'edge_type': self.edge_type,
                   'num_entities': self.num_entities,
                   'num_relations': self.num_relations,
                   'entity_pad_idx': self.entity_pad_idx,
                   'movie_idxs': self.movie_idxs,
                   'device': self.device}
        return kg_info
