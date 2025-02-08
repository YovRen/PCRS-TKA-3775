import json
from collections import defaultdict
import pickle as pkl
from tqdm.auto import tqdm


def get_item_set(file):
    entity = set()
    with open(file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = json.loads(line)
            for turn in line:
                for e in turn['movie_link']:
                    entity.add(e)
    return entity


def extract_subkg(kg, seed_set, n_hop):
    subkg = defaultdict(list)  # {head entity: [(relation, tail entity)]}
    subkg_hrt = set()  # {(head_entity, relation, tail_entity)}

    ripple_set = None
    for hop in range(n_hop):
        memories_h = set()  # [head_entity]
        memories_r = set()  # [relation]
        memories_t = set()  # [tail_entity]

        if hop == 0:
            tails_of_last_hop = seed_set  # [entity]
        else:
            tails_of_last_hop = ripple_set[2]  # [tail_entity]

        for entity in tqdm(tails_of_last_hop):
            for relation_and_tail in kg[entity]:
                h, r, t = entity, relation_and_tail[0], relation_and_tail[1]
                if (h, r, t) not in subkg_hrt:
                    subkg_hrt.add((h, r, t))
                    subkg[h].append((r, t))
                memories_h.add(h)
                memories_r.add(r)
                memories_t.add(t)

        ripple_set = (memories_h, memories_r, memories_t)

    return subkg


def kg2id(kg):
    entity_set = all_item

    with open('relation_set.json', encoding='utf-8') as f:
        relation_set = json.load(f)

    for head, relation_tails in tqdm(kg.items()):
        for relation_tail in relation_tails:
            if relation_tail[0] in relation_set:
                entity_set.add(head)
                entity_set.add(relation_tail[1])

    entity_url2entity_idx = {e: i for i, e in enumerate(entity_set)}
    print(f"# entity: {len(entity_url2entity_idx)}")
    relation_url2relation_idx = {r: i for i, r in enumerate(relation_set)}
    relation_url2relation_idx['self_loop'] = len(relation_url2relation_idx)
    print(f"# relation: {len(relation_url2relation_idx)}")

    kg_idx = {}
    for head, relation_tails in kg.items():
        if head in entity_url2entity_idx:
            head = entity_url2entity_idx[head]
            kg_idx[head] = [(relation_url2relation_idx['self_loop'], head)]
            for relation_tail in relation_tails:
                if relation_tail[0] in relation_url2relation_idx and relation_tail[1] in entity_url2entity_idx:
                    kg_idx[head].append((relation_url2relation_idx[relation_tail[0]], entity_url2entity_idx[relation_tail[1]]))

    return entity_url2entity_idx, relation_url2relation_idx, kg_idx


all_item = set()
file_list = [
    'train_data_raw.jsonl',
    'valid_data_raw.jsonl',
    'test_data_raw.jsonl',
]
for file in file_list:
    all_item |= get_item_set(file)
print(f'# all item: {len(all_item)}')

with open('../dbpedia/kg.pkl', 'rb') as f:
    kg = pkl.load(f)
subkg = extract_subkg(kg, all_item, 2)
entity_url2entity_idx, relation_url2relation_idx, subkg = kg2id(subkg)

with open('dbpedia_subkg.json', 'w', encoding='utf-8') as f:
    json.dump(subkg, f, ensure_ascii=False)
with open('entity_url2entity_idx.json', 'w', encoding='utf-8') as f:
    json.dump(entity_url2entity_idx, f, ensure_ascii=False)
with open('relation_url2relation_idx.json', 'w', encoding='utf-8') as f:
    json.dump(relation_url2relation_idx, f, ensure_ascii=False)
