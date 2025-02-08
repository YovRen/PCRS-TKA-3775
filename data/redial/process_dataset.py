import json
import re
import html
from collections import defaultdict
from copy import deepcopy
from tqdm.auto import tqdm
from transformers import AutoTokenizer
movie_pattern = re.compile(r'@\d+')


def process_utt(utt, movie_id2movie_name, replace):
    utt = ' '.join(utt.split())
    utt = html.unescape(utt)

    def convert(match):
        movie_id = match.group(0)[1:]
        if movie_id in movie_id2movie_name:
            movie_name = movie_id2movie_name[movie_id]
            if replace == "movie_name":
                return movie_name
            elif replace == "mask":
                return "<movie>"
            else:
                assert False
        else:
            return match.group(0)

    utt = re.sub(movie_pattern, convert, utt)
    return utt


def construct_prompt_tree(src_idxs, tokenizer, ary1, ary2):
    tree_text = ""
    for src_idx in src_idxs:
        src = ' '.join([relate.strip('()') for relate in entity_idx2entity_url[src_idx].strip('<>').split('/')[-1].split('_')])
        tree_text += src
        rel1_dict = deepcopy(two_level_tree.get(src, {}))
        ent12calc = defaultdict(int)
        for (rel1, ent1_dict) in rel1_dict.items():
            for (ent1, rel2_dict) in ent1_dict.items():
                ent12calc[ent1] = entity_calc_coarse[ent1]
        ent12calc = dict(sorted(ent12calc.items(), key=lambda x: x[1], reverse=True)[:ary1])
        keys_to_delete = []
        for (rel1, ent1_dict) in rel1_dict.items():
            for (ent1, rel2_dict) in list(ent1_dict.items()):  # 用 list() 包裹 items() 以避免修改期间迭代的问题
                if ent1 not in ent12calc:
                    keys_to_delete.append((rel1, ent1))
        for (rel1, ent1) in keys_to_delete:
            del rel1_dict[rel1][ent1]
        rel1s_to_delete = [rel1 for rel1, ent1_dict in rel1_dict.items() if len(ent1_dict) == 0]
        for rel1 in rel1s_to_delete:
            del rel1_dict[rel1]
        for (rel1, ent1_dict) in rel1_dict.items():
            tree_text += ' #' + rel1
            for (ent1, rel2_dict) in ent1_dict.items():
                tree_text += ' $' + ent1
                if len(rel2_dict) > 3:
                    print(ary1, ary2, len(rel1_dict), len(rel2_dict))
                rel2_dict = dict(sorted(rel2_dict.items(), key=lambda x: sum(entity_calc_coarse[key] for key in x[1]), reverse=True)[:ary2])
                for (rel2, ent2s) in rel2_dict.items():
                    tree_text += ' ##' + rel2
                    for ent2 in ent2s:
                        tree_text += ' $$' + ent2
        tree_text += tokenizer.sep_token
    return tree_text

def process(data_file, out_file, movie_set):
    with (open(data_file, 'r', encoding='utf-8') as fin, open(out_file, 'w', encoding='utf-8') as fout):
        for line in tqdm(f):
            dialog = json.loads(line)
            for i, message in enumerate(dialog['messages']):
                new_entity, new_entity_name = [], []
                for j, entity in enumerate(message['entity']):
                    if entity in entity_url2entity_idx:
                        new_entity.append(entity)
                        new_entity_name.append(message['entity_name'][j])
                dialog['messages'][i]['entity'] = new_entity
                dialog['messages'][i]['entity_name'] = new_entity_name
                new_movie, new_movie_name = [], []
                for j, movie in enumerate(message['movie']):
                    if movie in entity_url2entity_idx:
                        new_movie.append(movie)
                        new_movie_name.append(message['movie_name'][j])
                        # movies.add(movie)
                dialog['messages'][i]['movie'] = new_movie
                dialog['messages'][i]['movie_name'] = new_movie_name
            movie_id2movie_name = dialog['movieMentions']
            respondent_worker_id = dialog['respondentWorkerId']
            context_texts, context_entities = [], []
            messages = dialog['messages']
            turn_i = 0
            while turn_i < len(messages):
                sender_worker_id = messages[turn_i]['senderWorkerId']
                response_text = response_text_masked = 'System:' if sender_worker_id == respondent_worker_id else 'User:'
                response_entities, response_movies = [], []
                turn_j = turn_i
                while turn_j < len(messages) and messages[turn_j]['senderWorkerId'] == sender_worker_id:
                    text = process_utt(messages[turn_j]['text'], movie_id2movie_name, replace="movie_name")
                    text_masked = process_utt(messages[turn_j]['text'], movie_id2movie_name, replace="mask")
                    response_text += " " + text
                    response_text_masked += " " + text_masked
                    response_entities += [entity_url2entity_idx[entity] for entity in messages[turn_j]['movie'] + messages[turn_j]['entity'] if entity in entity_url2entity_idx]
                    response_movies += [entity_url2entity_idx[entity] for entity in messages[turn_j]['movie'] if entity in entity_url2entity_idx]
                    turn_j += 1
                response_entities = list(set(response_entities))
                context_response_entities = context_entities + response_entities
                context_response_entities_3_3_tree = deepcopy(construct_prompt_tree(context_response_entities, encoder_tokenizer, 3, 3))
                context_entities_3_3_tree = deepcopy(construct_prompt_tree(context_entities, encoder_tokenizer, 3, 3))

                context_response_entities_4_4_tree = deepcopy(construct_prompt_tree(context_response_entities, encoder_tokenizer, 4, 4))
                context_entities_4_4_tree = deepcopy(construct_prompt_tree(context_entities, encoder_tokenizer, 4, 4))

                context_response_entities_5_5_tree = deepcopy(construct_prompt_tree(context_response_entities, encoder_tokenizer, 5, 5))
                context_entities_5_5_tree = deepcopy(construct_prompt_tree(context_entities, encoder_tokenizer, 5, 5))

                context_response_entities_6_6_tree = deepcopy(construct_prompt_tree(context_response_entities, encoder_tokenizer, 6, 6))
                context_entities_6_6_tree = deepcopy(construct_prompt_tree(context_entities, encoder_tokenizer, 6, 6))

                context_response_entities_7_7_tree = deepcopy(construct_prompt_tree(context_response_entities, encoder_tokenizer, 7, 7))
                context_entities_7_7_tree = deepcopy(construct_prompt_tree(context_entities, encoder_tokenizer, 7, 7))

                context_response_entities_7_0_tree = deepcopy(construct_prompt_tree(context_response_entities, encoder_tokenizer, 7, 0))
                context_entities_7_0_tree = deepcopy(construct_prompt_tree(context_entities, encoder_tokenizer, 7, 0))

                context_response_entities_6_0_tree = deepcopy(construct_prompt_tree(context_response_entities, encoder_tokenizer, 6, 0))
                context_entities_6_0_tree = deepcopy(construct_prompt_tree(context_entities, encoder_tokenizer, 6, 0))

                context_response_entities_5_0_tree = deepcopy(construct_prompt_tree(context_response_entities, encoder_tokenizer, 5, 0))
                context_entities_5_0_tree = deepcopy(construct_prompt_tree(context_entities, encoder_tokenizer, 5, 0))

                context_response_entities_4_0_tree = deepcopy(construct_prompt_tree(context_response_entities, encoder_tokenizer, 4, 0))
                context_entities_4_0_tree = deepcopy(construct_prompt_tree(context_entities, encoder_tokenizer, 4, 0))

                context_response_entities_3_0_tree = deepcopy(construct_prompt_tree(context_response_entities, encoder_tokenizer, 3, 0))
                context_entities_3_0_tree = deepcopy(construct_prompt_tree(context_entities, encoder_tokenizer, 3, 0))

                turn = {
                    'context_texts': context_texts,
                    'response_text': response_text,
                    'response_text_masked': response_text_masked,
                    'context_entities': context_entities,
                    'context_response_entities': context_response_entities,
                    'response_entities': response_entities,
                    'response_movies': response_movies,
                    'context_response_entities_3_3_tree': context_response_entities_3_3_tree,
                    'context_entities_3_3_tree': context_entities_3_3_tree,
                    'context_response_entities_4_4_tree': context_response_entities_4_4_tree,
                    'context_entities_4_4_tree': context_entities_4_4_tree,
                    'context_response_entities_5_5_tree': context_response_entities_5_5_tree,
                    'context_entities_5_5_tree': context_entities_5_5_tree,
                    'context_response_entities_6_6_tree': context_response_entities_6_6_tree,
                    'context_entities_6_6_tree': context_entities_6_6_tree,
                    'context_response_entities_7_7_tree': context_response_entities_7_7_tree,
                    'context_entities_7_7_tree': context_entities_7_7_tree,
                    'context_response_entities_7_0_tree': context_response_entities_7_0_tree,
                    'context_entities_7_0_tree': context_entities_7_0_tree,
                    'context_response_entities_6_0_tree': context_response_entities_6_0_tree,
                    'context_entities_6_0_tree': context_entities_6_0_tree,
                    'context_response_entities_5_0_tree': context_response_entities_5_0_tree,
                    'context_entities_5_0_tree': context_entities_5_0_tree,
                    'context_response_entities_4_0_tree': context_response_entities_4_0_tree,
                    'context_entities_4_0_tree': context_entities_4_0_tree,
                    'context_response_entities_3_0_tree': context_response_entities_3_0_tree,
                    'context_entities_3_0_tree': context_entities_3_0_tree,
                }
                fout.write(json.dumps(turn, ensure_ascii=False) + '\n')
                movie_set |= set(response_movies)
                context_texts.append(response_text)
                context_entities.extend(response_entities)
                context_entities = list(set(context_entities))
                turn_i = turn_j


if __name__ == '__main__':
    dbpedia_subkg = json.load(open('dbpedia_subkg.json', 'r', encoding='utf-8'))
    entity_url2entity_idx = json.load(open('entity_url2entity_idx.json', 'r', encoding='utf-8'))
    entity_idx2entity_url = {value: key for key, value in entity_url2entity_idx.items()}
    relation_url2relation_idx = json.load(open('relation_url2relation_idx.json', 'r', encoding='utf-8'))
    relation_idx2relation_url = {value: key for key, value in relation_url2relation_idx.items()}
    entity_calc_coarse = json.load(open('entity_calc_coarse.json', encoding='utf-8'))
    two_level_tree = json.load(open('two_level_tree.json', encoding='utf-8'))
    encoder_tokenizer = AutoTokenizer.from_pretrained("../../model/roberta-base")
    encoder_tokenizer.add_special_tokens({'additional_special_tokens': ['<movie>'], })
    decoder_tokenizer = AutoTokenizer.from_pretrained("../../model/dialogpt-small")
    decoder_tokenizer.add_special_tokens({'pad_token': "<pad>", 'additional_special_tokens': ['<movie>'], })

    entity_calc_coarse = defaultdict(int)
    entity_calc_fine = defaultdict(lambda: defaultdict(int))
    one_level_tree = defaultdict(lambda: defaultdict(list))
    two_level_tree = defaultdict(lambda: defaultdict(dict))

    for src_idx in dbpedia_subkg:
        src_text = ' '.join([relate.strip('()') for relate in entity_idx2entity_url[int(src_idx)].strip('<>').split('/')[-1].split('_')])
        for (relation_idx, related_idx) in dbpedia_subkg[src_idx]:
            if int(relation_idx) == len(relation_idx2relation_url)-1:
                continue
            relation_text = ' '.join([relate.strip('()') for relate in relation_idx2relation_url[relation_idx].strip('<>').split('/')[-1].split('_')])
            related_text = ' '.join([relate.strip('()') for relate in entity_idx2entity_url[related_idx].strip('<>').split('/')[-1].split('_')])
            entity_calc_coarse[related_text] += 1
            entity_calc_fine[relation_text][related_text] += 1
            one_level_tree[src_text][relation_text].append(related_text)
    for src_idx in dbpedia_subkg:
        src_text = ' '.join([relate.strip('()') for relate in entity_idx2entity_url[int(src_idx)].strip('<>').split('/')[-1].split('_')])
        for (relation_idx, related_idx) in dbpedia_subkg[src_idx]:
            if int(relation_idx) == len(relation_idx2relation_url)-1:
                continue
            relation_text = ' '.join([relate.strip('()') for relate in relation_idx2relation_url[relation_idx].strip('<>').split('/')[-1].split('_')])
            related_text = ' '.join([relate.strip('()') for relate in entity_idx2entity_url[related_idx].strip('<>').split('/')[-1].split('_')])
            two_level_tree[src_text][relation_text][related_text] = one_level_tree[related_text]
    entity_calc_coarse = dict(sorted(entity_calc_coarse.items(), key=lambda x: x[1], reverse=True))
    entity_calc_fine = dict(sorted({relation_text: sorted(related_text2count.items(), key=lambda x: x[1], reverse=True) for relation_text, related_text2count in entity_calc_fine.items()}.items(), key=lambda x: sum([y[1] for y in x[1]]), reverse=True))

    movie_set = set()
    process('test_data_dbpedia.jsonl', 'test_data_processed.jsonl', movie_set)
    process('train_data_dbpedia.jsonl', 'train_data_processed.jsonl', movie_set)
    process('valid_data_dbpedia.jsonl', 'valid_data_processed.jsonl', movie_set)

    with open('movie_idxs.json', 'w', encoding='utf-8') as f:
        json.dump(list(movie_set), f, ensure_ascii=False)
    print(f'#movie: {len(movie_set)}')
    print("---------------------------Finished Preparing Cases------------------------------")