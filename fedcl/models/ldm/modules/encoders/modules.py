import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import CLIPTokenizer, CLIPTextModel
from functools import partial

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""
    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast  # TODO: add to reuquirements
        self.tokenizer = BertTokenizerFast.from_pretrained("/home/nlp/ct/projects/MOE-FedCL/PM/bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text

class TransformerWrapper(nn.Module):
    def __init__(self, num_tokens, max_seq_len, attn_layers, emb_dropout=0.0):
        super().__init__()
        self.num_tokens = num_tokens
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(num_tokens, 1280)  # BERT embedding dimension
        self.pos_emb = nn.Embedding(max_seq_len, 1280)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.attn_layers = attn_layers
        
    def forward(self, tokens, class_id, return_embeddings=False, embedding_manager=None):
        b, n = tokens.shape
        pos = torch.arange(n, device=tokens.device).unsqueeze(0).repeat(b, 1)
        
        x = self.token_emb(tokens)
        x = x + self.pos_emb(pos)
        x = self.emb_dropout(x)
        
        if embedding_manager is not None:
            x = embedding_manager(tokens, x, class_id)
        
        x = self.attn_layers(x)
        
        if return_embeddings:
            return x
        
        return x.mean(dim=1)  # Global average pooling

class Encoder(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=16, dim_feedforward=dim*4)
            for _ in range(depth)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizer model and add some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size=30522, max_seq_len=77,
                 device="cuda", use_tokenizer=True, embedding_dropout=0.0):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)

    def forward(self, text, class_id, embedding_manager=None):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)
        else:
            tokens = text
        z = self.transformer(tokens, class_id, return_embeddings=True, embedding_manager=embedding_manager)
        return z

    def encode(self, text, class_id, **kwargs):
        # output of length 77
        return self(text, class_id, **kwargs)

class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length

        def embedding_forward(
                self,
                input_ids = None,
                position_ids = None,
                inputs_embeds = None,
                embedding_manager = None,
            ) -> torch.Tensor:

                seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

                if position_ids is None:
                    position_ids = self.position_ids[:, :seq_length]

                if inputs_embeds is None:
                    inputs_embeds = self.token_embedding(input_ids)

                if embedding_manager is not None:
                    inputs_embeds = embedding_manager(input_ids, inputs_embeds)

                position_embeddings = self.position_embedding(position_ids)
                embeddings = inputs_embeds + position_embeddings
                
                return embeddings      

        self.transformer.text_model.embeddings.forward = embedding_forward.__get__(self.transformer.text_model.embeddings)

    def forward(self, text, class_id=None, embedding_manager=None):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                       return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        
        if embedding_manager is not None:
            z = self.transformer(input_ids=tokens, embedding_manager=embedding_manager)
        else:
            z = self.transformer(input_ids=tokens)
            
        return z

    def encode(self, text, class_id=None, **kwargs):
        return self(text, class_id, **kwargs)