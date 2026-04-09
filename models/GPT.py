import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import math
import torch.nn as nn

from models.utils import Transformer, RoPE

# dynamically select device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


"""
GPT
	A decoder-only multiheaded transformer with
	MLP vocabulary classifier.
"""
class GPT(nn.Module):
	"""
	GPT.__init__
		Constructs necessary internal modules for language
		model. Creates word embeddings if not provided,
		positional encoding, transformer layers, transformer,
		and linear classifier.

	Args:
		vocab_size: int size of vocab
		embed_dim: int embedding dimensioanlity
		num_layers: int number of transformer layers
		num_heads: int number of attention heads per layer
		word_emb: None or torch.Tensor of size (V, D)
		is_causal: boolean causal masking flag
	"""
	def __init__(
		self,
		vocab_size,
		embed_dim=256,
		num_layers=8,
		num_heads=8,
		word_emb=None,
		is_causal=True
	):
		super().__init__()
		assert embed_dim % num_heads == 0
		if word_emb is not None:
			self.token_embedding = nn.Embedding.from_pretrained(word_emb)
		else:
			self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)

		# positional encodings
		self.pos_enc = RoPE(embed_dim=embed_dim)

		# custom transformer
		self.transformer = Transformer(embed_dim, num_heads, num_layers)
		self.is_causal = is_causal

		# vocab classifier
		self.classifier = nn.Linear(in_features=embed_dim, out_features=vocab_size)
	

	"""
	GPT.forward
		Applies custom (provided) or learned embeddings across
		vocabulary. Positionally encodes embeddings. Generates
		causal mask and feeds embeddings through transformer.
		Classifies over vocabulary.

	Args:
		seq: torch.Tensor of size (B, N)

	Returns:
		torch.Tensor of size (B, N, V)
	"""
	def forward(self, seq):
		# embed sequence w poisitional encodings
		seq_embed = self.token_embedding(seq)
		seq_embed = self.pos_enc(seq_embed)
		seq_out = self.transformer(seq_embed, self.is_causal)

		# next token classification
		out = self.classifier(seq_out)
		return out