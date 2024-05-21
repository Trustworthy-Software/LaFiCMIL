import torch
from transformers import LongformerModel, BertModel, RobertaModel
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from nystrom_attention import NystromAttention


class TransLayer(nn.Module):
    
    def __init__(self, norm_layer=nn.LayerNorm, dim=512, dropout=0.1):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=dropout
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len=5, hidden_size=512, dropout_rate=0.1):
        super().__init__()

        self.pos_embed = nn.Embedding(max_len, hidden_size) # position embedding
        self.linear    = nn.Linear(hidden_size, hidden_size)

        self.norm = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)

        pos = pos.unsqueeze(0)
        pos = self.pos_embed(pos)

        pos = self.linear(pos)
        pos = torch.tanh(pos)

        e = x + pos
        return self.drop(self.norm(e))


class LaficMIL(torch.nn.Module):
    def __init__(self, num_labels, device, pos_type='PosEmb', dropout_rate=0.1, input_size=768, hidden_size=768):
        super(LaficMIL, self).__init__()
        assert pos_type in {'PosEmb', None}
        self.pos_type = pos_type
        self.num_labels = num_labels
        self.device = device
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.proj = False
        if not hidden_size == input_size:
            self.proj = True
            self._fc1 = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())
        self.category_vector = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.trans_layer = TransLayer(dim=hidden_size, dropout=dropout_rate)
        if pos_type == 'PosEmb':
            self.pos_emb = PositionalEmbedding(max_len=100, hidden_size=hidden_size, dropout_rate=dropout_rate)
        self.norm = nn.LayerNorm(hidden_size)
        self._fc2 = nn.Linear(hidden_size, self.num_labels)
    
    def get_bert_out(self, ids, masks, token_type_ids):
        _, bert_output = self.bert(ids.squeeze(0), attention_mask=masks.squeeze(0), token_type_ids=token_type_ids.squeeze(0), return_dict=False)
 
        return bert_output
    
    def forward(self, ids, masks, token_type_ids):
        bert_output = self.get_bert_out(ids, masks=masks, token_type_ids=token_type_ids)

        drop_output = self.dropout(bert_output)

        h = drop_output[None, :, :]
        if self.proj:
            h = self._fc1(h) #[B, n, 128]
        
        #---->category_vector
        B = h.shape[0]
        category_vectors = self.category_vector.expand(B, -1, -1).to(self.device)
        h = torch.cat((category_vectors, h), dim=1)

        #----> Positional Embedding
        if self.pos_type is not None:
            h = self.pos_emb(h)

        #---->Translayer
        h = self.trans_layer(h) #[B, N, 512]

        #---->category_vector
        h = self.norm(h)[:,0]
        logits = self._fc2(h) #[B, n_classes]
        
        return logits


class BERTPlus(torch.nn.Module):
    def __init__(self, dropout_rate, num_labels):
        super(BERTPlus, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.classifier = torch.nn.Linear(768*2, num_labels)

    def forward(self, ids, mask, token_type_ids):
        _, truncated_output = self.bert(ids[:, 0,:], attention_mask=mask[:, 0,:], token_type_ids=token_type_ids[:, 0,:], return_dict=False)
        _, additional_text_output = self.bert(ids[:, 1,:], attention_mask=mask[:, 1,:], token_type_ids=token_type_ids[:, 1,:], return_dict=False)
        concat_output = torch.cat((truncated_output, additional_text_output), dim=1) # batch_size, 768*2
        drop_output = self.dropout(concat_output) # batch_size, 768*2
        logits = self.classifier(drop_output) # batch_size, num_labels
        return logits

        
class BERTClass(torch.nn.Module):
    def __init__(self, dropout_rate, num_labels):
        super(BERTClass, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.classifier = torch.nn.Linear(768, num_labels)

    def forward(self, ids, mask, token_type_ids):
        _, bert_output = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        drop_output = self.dropout(bert_output)
        logits = self.classifier(drop_output)
        return logits

class LongformerClass(torch.nn.Module):
    def __init__(self, num_labels):
        super(LongformerClass, self).__init__()
        self.longformer = LongformerModel.from_pretrained('allenai/longformer-base-4096', add_pooling_layer=False,
                                                  gradient_checkpointing=True)
        self.classifier = LongformerClassificationHead(hidden_size=768, hidden_dropout_prob=0.1, num_labels=num_labels)

    def forward(self, ids, mask, token_type_ids):
        # Initialize global attention on CLS token
        global_attention_mask = torch.zeros_like(ids)
        global_attention_mask[:, 0] = 1
        sequence_output, _ = self.longformer(ids, attention_mask=mask, global_attention_mask=global_attention_mask,
                            token_type_ids=token_type_ids, return_dict=False)
        logits = self.classifier(sequence_output)
        return logits

class LongformerClassificationHead(torch.nn.Module):
    # This class is from https://huggingface.co/transformers/_modules/transformers/models/longformer
    # /modeling_longformer.html#LongformerForSequenceClassification
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_size, hidden_dropout_prob, num_labels): # config from transformers.LongformerConfig.from_pretrained('allenai/longformer-base-4096')
        super().__init__()
        self.dense = torch.nn.Linear(hidden_size, hidden_size)
        self.dropout = torch.nn.Dropout(hidden_dropout_prob)
        self.out_proj = torch.nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states, **kwargs):
        hidden_states = hidden_states[:, 0, :]  # take <s> token (equiv. to [CLS])
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output = self.out_proj(hidden_states)
        return output

class ToBERTModel(torch.nn.Module):
    def __init__(self, num_labels, device):
        super(ToBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.trans = torch.nn.TransformerEncoderLayer(d_model=768, nhead=2)
        self.fc = torch.nn.Linear(768, 30)
        self.classifier = torch.nn.Linear(30, num_labels)
        self.device = device

    def forward(self, ids, mask, token_type_ids, length):
        _, pooled_out = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)

        chunks_emb = pooled_out.split_with_sizes(length)
        batch_emb_pad = torch.nn.utils.rnn.pad_sequence(
            chunks_emb, padding_value=0, batch_first=True)
        batch_emb = batch_emb_pad.transpose(0, 1)  # (B,L,D) -> (L,B,D)
        padding_mask = np.zeros([batch_emb.shape[1], batch_emb.shape[0]]) # Batch size, Sequence length
        for idx in range(len(padding_mask)):
            padding_mask[idx][length[idx]:] = 1 # padding key = 1 ignored

        padding_mask = torch.tensor(padding_mask).to(self.device, dtype=torch.bool)
        trans_output = self.trans(batch_emb, src_key_padding_mask=padding_mask)
        mean_pool = torch.mean(trans_output, dim=0) # Batch size, 768
        fc_output = self.fc(mean_pool)
        relu_output = F.relu(fc_output)
        logits = self.classifier(relu_output)

        return logits

if __name__ == "__main__":

    pass
