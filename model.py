import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn
from layers import *

class BERT4Rec(nn.Module):
    def __init__(self, usernum, itemnum, args, config, embeddings=None):
        super(BERT4Rec, self).__init__()
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.inner_size = config["inner_size"]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.n_items = itemnum
        self.max_seq_length = args.maxlen
        self.dev = args.device
        
        if embeddings is None:
            self.item_emb = torch.nn.Embedding(self.n_items +1, self.hidden_size, padding_idx=0)
            torch.nn.init.xavier_normal_(self.item_emb.weight)
        else:
            self.item_emb = torch.nn.Embedding(embeddings.shape[0], embeddings.shape[1], padding_idx=0, _weight=embeddings)
           
        self.position_emb = nn.Embedding(
            self.max_seq_length, self.hidden_size
        )  # add mask_token at the last
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.output_ffn = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_gelu = nn.GELU()
        self.output_ln = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.output_bias = nn.Parameter(torch.zeros(self.n_items))
    
    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = item_seq != 0
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(
                extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1))
            )
        extended_attention_mask = torch.where(extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask
    
    def log2feats(self, log_seqs):
        log_seqs = torch.LongTensor(log_seqs).to(self.dev)
        position_ids = torch.arange(
            log_seqs.size(1), dtype=torch.long, device=log_seqs.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(log_seqs)
        position_embedding = self.position_emb(position_ids)
        item_embedding = self.item_emb(log_seqs)
        input_emb = item_embedding + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        extended_attention_mask = self.get_attention_mask(log_seqs, bidirectional=True)
        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        ffn_output = self.output_ffn(trm_output[-1])
        ffn_output = self.output_gelu(ffn_output)
        output = self.output_ln(ffn_output)
        return output
    
    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training        
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices=None,is_eavl=False): # for inference
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        if not is_eavl: return final_feat

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)
        
class SASRec(nn.Module):
    def __init__(self, user_num, item_num, args, config, embeddings=None):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.embedding_dim = config['embedding_dim']
        self.num_heads = config['num_heads']
        self.num_blocks = config['num_blocks']
        self.dropout_rate = config['dropout_rate']
        self.layer_norm_eps = config['layer_norm_eps']
        self.maxlen = args.maxlen
        self.dev = args.device
        self.fwd_layer = None
        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        if embeddings is None:
            self.item_emb = torch.nn.Embedding(self.item_num+1, self.embedding_dim, padding_idx=0)
            torch.nn.init.xavier_normal_(self.item_emb.weight)
        else:
            self.item_emb = torch.nn.Embedding(embeddings.shape[0], embeddings.shape[1], padding_idx=0, _weight=embeddings)

        self.pos_emb = torch.nn.Embedding(self.maxlen, self.embedding_dim) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=self.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(self.embedding_dim, eps=self.layer_norm_eps)

        for _ in range(self.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(self.embedding_dim, eps=self.layer_norm_eps)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(self.embedding_dim, self.num_heads, self.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(self.embedding_dim, eps=self.layer_norm_eps)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(self.embedding_dim, self.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training        
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices=None, is_eavl=False): # for inference
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        if not is_eavl : return final_feat

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)

class GRU4Rec(nn.Module):
    def __init__(self, user_num, item_num, args, config, embeddings=None):
        super(GRU4Rec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']
        self.output_size = config['output_size']
        self.dropout_rate = config['dropout_rate']
        self.num_layers = config['num_layers']
        self.final_activation = nn.Tanh()
        self.dev = args.device
        
        
        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        if embeddings is None:
            self.item_emb = torch.nn.Embedding(self.item_num+1, self.input_size, padding_idx=0)
            torch.nn.init.xavier_normal_(self.item_emb.weight)
        else:
            self.item_emb = torch.nn.Embedding(embeddings.shape[0], embeddings.shape[1], padding_idx=0, _weight=embeddings)

        self.emb_dropout = torch.nn.Dropout(p=self.dropout_rate)
        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, dropout=self.dropout_rate,batch_first=True)
        self.h2o = nn.Linear(self.hidden_size, self.output_size)
        

    def log2feats(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        h0 = torch.zeros(self.num_layers, log_seqs.shape[0], self.hidden_size).to(self.dev)
        seqs = self.emb_dropout(seqs)
        output, _ = self.gru(seqs, h0)
        log_feats = self.final_activation(self.h2o(output))
       
        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training        
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices=None, is_eavl=False): # for inference
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        if not is_eavl: return final_feat

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)

class LANE(nn.Module):
    def __init__(self,usernum, itemnum, inte_model, args):
        super(LANE, self).__init__()
        self.input_dim = args.embedding_dim
        self.hidden = args.hidden
        self.num_heads = args.num_heads
        self.dev = args.device

        self.linear_q = nn.Linear(self.input_dim , self.hidden, bias=False)
        self.linear_k = nn.Linear(self.input_dim , self.hidden, bias=False)
        self.linear_v = nn.Linear(self.input_dim , self.hidden, bias=False)
        self.linear_output = nn.Linear(self.hidden,self.input_dim ,bias=False)
        self.dropout = nn.Dropout(args.dropout_rate)
        self._norm_fact = 1 / np.sqrt(self.hidden // self.num_heads)
        self.layer_norm1 = nn.LayerNorm(self.input_dim)
        self.layer_norm2 = nn.LayerNorm(self.input_dim)
        self.inte_model = inte_model
        self.inte_model.dev = args.device
        self.fwd_layer = PointWiseFeedForward(self.input_dim, args.dropout_rate)
        
        # embeddings = self.inte_model.item_emb.weight
        # self.item_emb = torch.nn.Embedding(embeddings.shape[0], embeddings.shape[1], padding_idx=0, _weight=embeddings)
        
    def attention_and_Feed_Forwar(self, querys, keys, values):

        batch = querys.shape[0]

        qn = querys.shape[1]
        kn = keys.shape[1]
        vn = values.shape[1]

        nh = self.num_heads
        d = self.hidden // nh  # hidden of each head

        q = self.linear_q(querys).reshape(batch, qn, nh, d).transpose(1, 2)  # (batch, nh, qn, d)
        k = self.linear_k(keys).reshape(batch, kn, nh, d).transpose(1, 2)  # (batch, nh, kn, d)
        v = self.linear_v(values).reshape(batch, vn, nh, d).transpose(1, 2)  # (batch, nh, vn, d)

        scores = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, qn, kn
        attention_weights = torch.softmax(scores, dim=-1)  # batch, nh, qn, kn
        attention_weights = self.dropout(attention_weights)

        att = torch.matmul(attention_weights, v)  # batch, nh, qn, d
        att = att.transpose(1, 2).reshape(batch, qn, self.hidden)  # batch, qn, hidden
        att_feats = self.layer_norm1(self.linear_output(att) + querys)
        att_feats = self.layer_norm2(self.fwd_layer(att_feats))

        return att_feats
    
    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs, k_and_v,has_tag=False):
        if has_tag:
            querys = self.inte_model.log2feats(log_seqs,pos_seqs[:,-1])        
        else:
            querys = self.inte_model.log2feats(log_seqs)
        k_and_v = torch.stack(k_and_v, dim=0).to(self.dev)
        att_feats = self.attention_and_Feed_Forwar(querys,k_and_v,k_and_v)
        
        pos_embs = self.inte_model.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.inte_model.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (att_feats * pos_embs).sum(dim=-1)
        neg_logits = (att_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits

    def predict(self,user_ids, log_seqs, k_and_v, candidates,has_tag=False):
        # x: tensor of shape (batch, n, input_dim)
        if has_tag:
            querys = self.inte_model.predict(user_ids,log_seqs,candidates).unsqueeze(1)     
        else:
            querys = self.inte_model.predict(user_ids,log_seqs).unsqueeze(1)
        k_and_v = torch.stack(k_and_v, dim=0).to(self.dev)
        candidates = torch.LongTensor(candidates).to(self.dev)

        att_feats = self.attention_and_Feed_Forwar(querys, k_and_v, k_and_v)
        item_embs = self.inte_model.item_emb(candidates)
        logits = torch.bmm(item_embs,att_feats.permute(0,2,1)).squeeze(-1)
        return logits  #, attention_weights[:, 0, 0, :]

    def get_weight(self,user_ids,log_seqs,key, tag, has_tag=False):
        if has_tag:
            query = self.inte_model.predict(user_ids,log_seqs,tag).unsqueeze(1)
        else:
            query = self.inte_model.predict(user_ids,log_seqs).unsqueeze(1)
        key = torch.stack(key, dim=0).to(self.dev)
        q = self.linear_q(query)
        k = self.linear_k(key)
        scores = torch.matmul(q, k.transpose(1, 2)).squeeze(1) / np.sqrt(self.hidden)
        attention_weights = torch.softmax(scores, dim=-1)
        return attention_weights

