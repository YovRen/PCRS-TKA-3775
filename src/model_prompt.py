import os
import math
import re
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import RGCNConv


class PreTrainedModel(nn.Module):

    def __init__(self, *inputs, **kwargs):
        super(PreTrainedModel, self).__init__()

    def save_model(self, output_dir, old_tag, mode):
        def update_tag(string, mode):
            matches = re.findall(r'(\d+(?:rec_pre|gen_pre|gen|rec))', string)
            if not matches:
                return string + '1' + mode
            last_match = matches[-1]
            value_match = re.search(r'(\d+)', last_match)
            last_value = int(value_match.group(1))
            last_occurrence = last_match.replace(value_match.group(0), '')
            if last_occurrence == mode:
                last_index = string.rfind(last_match)
                new_value = str(last_value + 1)
                return string[:last_index] + new_value + last_occurrence
            else:
                return string + '1' + mode

        state_dict = {k: v for k, v in self.state_dict().items() if 'edge' not in k}
        new_tag = update_tag(old_tag, mode)
        save_path = os.path.join(output_dir, new_tag + '_model.pt')
        torch.save(state_dict, save_path)
        print(f"-------------model {new_tag} saved to  {output_dir}/{new_tag}_model.pt--------------")
        return new_tag

    def load_model(self, output_dir, old_tag):
        load_path = os.path.join(output_dir, old_tag + '_model.pt')
        missing_keys, unexpected_keys = self.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')), strict=False)
        print(missing_keys, unexpected_keys)
        print(f"-------------model {old_tag} loaded from {output_dir}/{old_tag}_model.pt--------------")


class SelfAttentionLayerBatch(nn.Module):
    def __init__(self, dim, da, alpha=0.2, dropout=0.5):
        super().__init__()
        self.dim = dim
        self.da = da
        self.alpha = alpha
        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(self.dim, self.da)))
        self.b = nn.Parameter(torch.zeros(size=(self.da, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.xavier_uniform_(self.b.data, gain=1.414)

    def forward(self, x, mask):
        assert self.dim == x.shape[2]
        mask = 1e-30 * mask.float()
        e = torch.matmul(torch.tanh(torch.matmul(x, self.a)), self.b)
        attention = F.softmax(e + mask.unsqueeze(-1), dim=1)
        return torch.matmul(torch.transpose(attention, 1, 2), x).squeeze(1)


class ErniePromptModel(PreTrainedModel):

    def __init__(self, kg_info, add_user=False, add_align=False, add_tree=False, n_prefix_rec=10, n_prefix_conv=20, token_hidden_size=768, 
                 decoder_hidden_size=768, decoder_num_attention_heads=12, decoder_num_layers=12, decoder_num_blocks=2):
        super(ErniePromptModel, self).__init__()

        self.token_hidden_size = token_hidden_size
        self.entity_hidden_size = self.decoder_hidden_size // 2
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_num_attention_heads = decoder_num_attention_heads
        self.decoder_head_dim = self.decoder_hidden_size // self.decoder_num_attention_heads
        self.decoder_num_layers = decoder_num_layers
        self.decoder_num_blocks = decoder_num_blocks
        self.n_prefix_rec = n_prefix_rec
        self.n_prefix_conv = n_prefix_conv
        self.add_user = add_user
        self.add_tree = add_tree
        self.add_align = add_align

        self.kg_encoder = RGCNConv(self.entity_hidden_size, self.entity_hidden_size, num_relations=kg_info['num_relations'], num_bases=8)
        self.node_embeds = nn.Parameter(torch.empty(kg_info['num_entities'], self.entity_hidden_size))
        stdv = math.sqrt(6.0 / (self.node_embeds.size(-2) + self.node_embeds.size(-1)))
        self.node_embeds.data.uniform_(-stdv, stdv)
        self.edge_index = nn.Parameter(kg_info['edge_index'], requires_grad=False)
        self.edge_type = nn.Parameter(kg_info['edge_type'], requires_grad=False)
        self.entity_pad_idx = kg_info['entity_pad_idx']

        self.entity_proj1 = nn.Sequential(nn.Linear(self.entity_hidden_size, self.entity_hidden_size // 2), nn.ReLU(), nn.Linear(self.entity_hidden_size // 2, self.entity_hidden_size), )
        self.entity_proj2 = nn.Linear(self.entity_hidden_size, self.decoder_hidden_size)

        self.token_proj1 = nn.Sequential(nn.Linear(self.token_hidden_size, self.token_hidden_size // 2), nn.ReLU(), nn.Linear(self.token_hidden_size // 2, self.token_hidden_size), )
        self.token_proj2 = nn.Linear(self.token_hidden_size, self.decoder_hidden_size)

        self.cross_attn = nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size, bias=False)  # token@entity

        self.prompt_proj1 = nn.Sequential(nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size // 2), nn.ReLU(), nn.Linear(self.decoder_hidden_size // 2, self.decoder_hidden_size), )
        self.prompt_proj2 = nn.Linear(self.decoder_hidden_size, self.decoder_num_layers * self.decoder_num_blocks * self.decoder_hidden_size)

        if self.add_user:
            self.intent_attn = SelfAttentionLayerBatch(self.token_hidden_size, self.token_hidden_size)
            self.intent_proj = nn.Linear(self.token_hidden_size, self.decoder_hidden_size)
            self.cross_attn_2 = nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size, bias=False)  # intent@entity

        if self.add_tree:
            self.tree_proj1 = nn.Sequential(nn.Linear(self.token_hidden_size, self.token_hidden_size // 2), nn.ReLU(), nn.Linear(self.token_hidden_size // 2, self.token_hidden_size), )
            self.tree_proj2 = nn.Linear(self.token_hidden_size, self.decoder_hidden_size)
            self.cross_attn_3 = nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size, bias=False)  # tree@entity
            self.cross_attn_4 = nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size, bias=False)  # tree@entity

        if self.add_align:
            self.entity_align_attn = SelfAttentionLayerBatch(self.decoder_hidden_size, self.decoder_hidden_size)
            self.entity_align_proj = nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size)
            self.token_align_attn = SelfAttentionLayerBatch(self.decoder_hidden_size, self.decoder_hidden_size)
            self.token_align_proj = nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size)
            if self.add_tree:
                self.tree_align_attn = SelfAttentionLayerBatch(self.decoder_hidden_size, self.decoder_hidden_size)
                self.tree_align_proj = nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size)
            self.rank_net = nn.Sequential(nn.Linear(self.decoder_hidden_size * 2, self.decoder_hidden_size), nn.ReLU(), nn.Linear(self.decoder_hidden_size, 1))
            self.bce_loss = nn.BCELoss()

        if self.n_prefix_rec is not None:
            self.rec_prefix_embeds = nn.Parameter(torch.empty(self.n_prefix_rec, self.decoder_hidden_size))
            nn.init.normal_(self.rec_prefix_embeds)
            self.rec_prefix_proj = nn.Sequential(nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size // 2), nn.ReLU(), nn.Linear(self.decoder_hidden_size // 2, self.decoder_hidden_size))
        if self.n_prefix_conv is not None:
            self.conv_prefix_embeds = nn.Parameter(torch.empty(self.n_prefix_conv, self.decoder_hidden_size))
            nn.init.normal_(self.conv_prefix_embeds)
            self.conv_prefix_proj = nn.Sequential(nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size // 2), nn.ReLU(), nn.Linear(self.decoder_hidden_size // 2, self.decoder_hidden_size))

    def get_entity_embeds(self):
        node_embeds = self.node_embeds
        entity_embeds = self.kg_encoder(node_embeds, self.edge_index, self.edge_type) + node_embeds
        entity_embeds = self.entity_proj1(entity_embeds) + entity_embeds
        entity_embeds = self.entity_proj2(entity_embeds)
        return entity_embeds

    def forward(self, token_input_ids=None, tree_input_ids=None, token_embeds=None, tree_embeds=None, rec_labels=None, entity_ids=None, sup_embeds=None, sup_score=None, output_entity=False, use_rec_prefix=False, use_conv_prefix=False):

        batch_size, entity_len = entity_ids.shape[:2]
        node_embeds = self.get_entity_embeds()
        entity_embeds = node_embeds[entity_ids]

        batch_size, token_len = token_embeds.shape[:2]
        token_embeds = self.token_proj1(token_embeds) + token_embeds  # (batch_size, token_len, decoder_hidden_size)
        token_embeds = self.token_proj2(token_embeds)

        attn_weights = self.cross_attn(token_embeds) @ entity_embeds.permute(0, 2, 1)  # (batch_size, token_len, entity_len)
        attn_weights /= self.decoder_hidden_size
        if output_entity:
            token2entity_weights = F.softmax(attn_weights, dim=1).permute(0, 2, 1)
            prompt_embeds = token2entity_weights @ token_embeds + entity_embeds  # (batch_size, entity_len, decoder_hidden_size)
            prompt_len = entity_len
        else:
            entity2token_weights = F.softmax(attn_weights, dim=2)
            prompt_embeds = entity2token_weights @ entity_embeds + token_embeds  # (batch_size, token_len, decoder_hidden_size)
            prompt_len = token_len

        user_loss = None
        if self.add_user:
            intent_ori_embeds = torch.cat([token_embeds, entity_embeds], dim=1)
            intent_ori_mask = torch.cat([token_input_ids == 1, entity_ids == self.entity_pad_idx], dim=1)
            intent_embeds = self.intent_attn(intent_ori_embeds, intent_ori_mask)
            intent_embeds = self.intent_proj(intent_embeds) + intent_embeds
            intent_embeds = intent_embeds.unsqueeze(1)   # (batch_size, 1, hidden_size)
            intent_rec_logits = intent_embeds.squeeze(1) @ node_embeds.T
            user_loss = torch.mean(F.cross_entropy(intent_rec_logits, rec_labels, reduce=False) * (rec_labels != self.entity_pad_idx))
            entity2intent_weights = self.cross_attn_2(intent_embeds) @ entity_embeds.permute(0, 2, 1)  # (batch_size, 1, entity_len)
            entity2intent_weights /= self.decoder_hidden_size
            entity2intent_weights = F.softmax(entity2intent_weights, dim=1)
            intent_embeds = entity2intent_weights @ entity_embeds + intent_embeds  # (batch_size, 1, entity_len)) # (batch_size, entity_len, hidden_size)
            prompt_embeds = torch.cat([prompt_embeds, intent_embeds], dim=1)
            prompt_len += 1

        if self.add_tree:
            batch_size, tree_len = tree_embeds.shape[:2]
            tree_embeds = self.tree_proj1(tree_embeds) + tree_embeds  # (batch_size, token_len, decoder_hidden_size)
            tree_embeds = self.tree_proj2(tree_embeds)
            if output_entity:
                entity2tree_weights = self.cross_attn_3(tree_embeds) @ entity_embeds.permute(0, 2, 1)  # (batch_size, token_len, entity_len)
                entity2tree_weights /= self.decoder_hidden_size
                entity2tree_weights = F.softmax(entity2tree_weights, dim=1)
                tree_embeds = entity2tree_weights @ entity_embeds + tree_embeds  # (batch_size, entity_len, decoder_hidden_size)
                prompt_embeds = torch.cat([prompt_embeds, tree_embeds], dim=1)
                prompt_len += tree_len
            else:
                entity2tree_weights = self.cross_attn_4(tree_embeds) @ token_embeds.permute(0, 2, 1)  # (batch_size, token_len, entity_len)
                entity2tree_weights /= self.decoder_hidden_size
                entity2tree_weights = F.softmax(entity2tree_weights, dim=1)
                tree_embeds = entity2tree_weights @ token_embeds + tree_embeds  # (batch_size, entity_len, decoder_hidden_size)
                prompt_embeds = torch.cat([prompt_embeds, tree_embeds], dim=1)
                prompt_len += tree_len

        align_loss = None
        if self.add_align:
            if not output_entity:
                rec_labels = torch.arange(batch_size).cuda()
            entity_align_attn_embed = self.entity_align_attn(entity_embeds, entity_ids == self.entity_pad_idx)  # (batch_size, 1, hidden_size)
            entity_align_attn_embed = self.entity_align_proj(entity_align_attn_embed) + entity_align_attn_embed
            token_align_attn_embed = self.token_align_attn(token_embeds, token_input_ids == 1)  # (batch_size, 1, hidden_size)
            token_align_attn_embed = self.token_align_proj(token_align_attn_embed) + token_align_attn_embed
            similarity_matrix = self.rank_net(torch.cat([entity_align_attn_embed.unsqueeze(1).repeat(1, batch_size, 1), token_align_attn_embed.unsqueeze(0).repeat(batch_size, 1, 1)], dim=2)).squeeze()  # input_j label_i sim
            similarity_matrix = torch.diag(similarity_matrix).unsqueeze(1).repeat(1, batch_size) - similarity_matrix
            label_pairs_mask = torch.eq(rec_labels.unsqueeze(1), rec_labels.unsqueeze(0)).float() / 2
            align_loss = self.bce_loss(F.sigmoid(similarity_matrix), label_pairs_mask)
            if self.add_tree:  # (batch_size, 1, hidden_size)
                tree_align_attn_embeds = self.tree_align_attn(tree_embeds, tree_input_ids == 1)
                tree_align_attn_embeds = self.tree_align_proj(tree_align_attn_embeds) + tree_align_attn_embeds
                similarity_matrix = self.rank_net(torch.cat([entity_align_attn_embed.unsqueeze(1).repeat(1, batch_size, 1), tree_align_attn_embeds.unsqueeze(0).repeat(batch_size, 1, 1)], dim=2)).squeeze()  # input_j label_i sim
                similarity_matrix = torch.diag(similarity_matrix).unsqueeze(1).repeat(1, batch_size) - similarity_matrix
                align_loss += self.bce_loss(F.sigmoid(similarity_matrix), label_pairs_mask)

        if self.n_prefix_rec is not None and use_rec_prefix:
            prefix_embeds = self.rec_prefix_proj(self.rec_prefix_embeds) + self.rec_prefix_embeds
            prefix_embeds = prefix_embeds.expand(prompt_embeds.shape[0], -1, -1)
            prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            prompt_len += self.n_prefix_rec
        if self.n_prefix_conv is not None and use_conv_prefix:
            prefix_embeds = self.conv_prefix_proj(self.conv_prefix_embeds) + self.conv_prefix_embeds
            prefix_embeds = prefix_embeds.expand(prompt_embeds.shape[0], -1, -1)
            prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            prompt_len += self.n_prefix_conv
        prompt_embeds = self.prompt_proj1(prompt_embeds) + prompt_embeds
        prompt_embeds = self.prompt_proj2(prompt_embeds)
        prompt_embeds = prompt_embeds.reshape(batch_size, prompt_len, self.decoder_num_layers, self.decoder_num_blocks, self.decoder_num_attention_heads, self.decoder_head_dim).permute(2, 3, 0, 4, 1, 5)  # (n_layer, n_block, batch_size, n_head, prompt_len, head_dim)

        return prompt_embeds, user_loss, align_loss
