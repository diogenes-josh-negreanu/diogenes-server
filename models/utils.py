import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

# dynamically select device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


"""
RoPE
	Applies positional encoding to sequential word embeddings
	via rotary positional encoding.
"""
class RoPE(nn.Module):
	"""
	RoPE.__init__
		Initializes encoding with proper embedding dimension.

	Args:
		embed_dim: int embedding dimensionality
	"""
	def __init__(self, embed_dim):
		super().__init__()
		self.embed_dim = embed_dim

	"""
	RoPE.forward
		Applies rotary positional encoding to input.
	
	Args:
		x: torch.Tensor of size (B, N, D)

	Returns:
		torch.Tensor of size (B, N, D)
	"""
	def forward(self, x):
		seq_len = x.shape[1]

		theta = 1.0 / (10000 ** (torch.arange(0, self.embed_dim, 2).float() / self.embed_dim)).to(x.device)

		seq_idx = torch.arange(seq_len).float().to(x.device)

		idx_theta = torch.einsum("i,j->ij", seq_idx, theta)
		hat_theta = torch.cat([idx_theta, idx_theta], axis=-1)
		sin = torch.sin(hat_theta)
		cos = torch.cos(hat_theta)
		xu, xd = x[..., : self.embed_dim // 2], x[..., self.embed_dim // 2 :]
		hat_x = torch.cat([-xd, xu], axis=-1)
		return x * cos + hat_x * sin


"""
SinusoidalEncoding
	Applies positional encoding to sequential word embeddings
	via sinusoidal encoding.
"""
class SinusoidalEncoding(nn.Module):
	"""
	SinusoidalEncoding.__init__
		Initializes encoding with proper embedding dimension.

	Args:
		embed_dim: int embedding dimensionality
	"""
	def __init__(self, embed_dim):
		super().__init__()
		self.embed_dim = embed_dim

	"""
	SinusoidalEncoding.forward
		Applies sinusoidal positional encoding to input.
	
	Args:
		x: torch.Tensor of size (B, N, D)

	Returns:
		torch.Tensor of size (B, N, D)
	"""
	def forward(self, x):
		batch_size = x.shape[0]
		seq_len = x.shape[1]

		# calcualte sinusoidal encodings
		pe = torch.zeros(1, seq_len, self.embed_dim).to(x.device)
		pos = torch.arange(0, seq_len, dtype=torch.float32)
		enc = torch.exp((-math.log(10000.0)) * (torch.arange(0, self.embed_dim, step=2, dtype=torch.float32) / self.embed_dim))

		# calculate positional encoding
		prod = torch.outer(pos, enc)
		pe[0, :, 0::2] = torch.sin(prod)
		pe[0, :, 1::2] = torch.cos(prod)
		pe = pe.expand(batch_size, -1, -1)

		# apply as residual
		x = x + pe
		return x


"""
MultiheadAttention
	Multi-headed attention with/without causal
	masking applied.
"""
class MultiheadAttention(nn.Module):
	"""
	MultiheadAttention.__init__
		Constructs key, query, and value matrices, and
		final linear layer.
	
	Args:
		emb_dim: int size of embedding dimension
		num_heads: int number of attention heads
	"""
	def __init__(self, emb_dim, num_heads):
		super().__init__()

		assert emb_dim % num_heads == 0
		self.emb_dim = emb_dim
		self.head_dim = int(emb_dim / num_heads)
		self.num_heads = num_heads
		
		# set up key, query, and value linear transformations
		self.q_linear = nn.Linear(emb_dim, emb_dim)
		self.k_linear = nn.Linear(emb_dim, emb_dim)
		self.v_linear = nn.Linear(emb_dim, emb_dim)

		self.concat_linear = nn.Linear(emb_dim, emb_dim)


	"""
	MultiheadAttention.scaled_dot_product_attention
		Applies scaled dot product attention to input
		previously passed through key, query, and value
		transformations.
	
	Args:
		q: torch.Tensor input queries
		k: torch.Tensor input keys
		v: torch.Tensor input values
		is_causal: boolean causal masking flag
	
	Returns:
		torch.Tensor of size (B, N, D)
	"""
	def scaled_dot_product_attention(self, q, k, v, is_causal):
		seq_len = q.shape[1]

		# dot product self attention
		q = q.transpose(1, 2)
		k = k.transpose(1, 2)
		v = v.transpose(1, 2)
		dots = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

		# apply causal mask if causal
		if is_causal:
			mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
			causal_mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to(q.device)
			dots = dots + causal_mask
		
		attn = F.softmax(dots, dim=-1)
		return torch.matmul(attn, v).transpose(1, 2).contiguous()

	
	"""
	MultiheadAttention.forward
		Runs a forward pass through multiheaded attention
		layer. Splits input dimensions across heads,
		runs through query, key, and value transformations,
		applies scaled dot product attention, concatenates
		and passes through a final linear layer.
	
	Args:
		x: torch.Tensor of size (B, N, D)
		is_causal: boolean causal masking flag
	
	Returns:
		torch.Tensor of size (B, N, D)
	"""
	def forward(self, x, is_causal):
		bs = x.shape[0]

		# run through query, key, and value transformations
		q = self.q_linear(x).view(bs, -1, self.num_heads, self.head_dim)
		k = self.k_linear(x).view(bs, -1, self.num_heads, self.head_dim)
		v = self.v_linear(x).view(bs, -1, self.num_heads, self.head_dim)

		# calculate attentions, concatenate multiple heads
		attn = self.scaled_dot_product_attention(q, k, v, is_causal)
		attn = attn.reshape(bs, -1, self.emb_dim)
		return self.concat_linear(attn)


"""
TransformerLayer
	Individual transformer layer employing
	multiheaded attention and a feed forward.
"""
class TransformerLayer(nn.Module):
	"""
	TransformerLayer.__init__
		Configures internal multiheaded attention
		layer, feed forward layer, and batch
		normalization.
	
	Args:
		emb_dim: int embedding dimension
		num_head: int number of heads
	"""
	def __init__(self, emb_dim, num_heads):
		super().__init__()
		# self.attn_layer = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
		self.attn_layer = MultiheadAttention(emb_dim, num_heads)

		self.feed_forward = nn.Sequential(
			nn.Linear(emb_dim, 3072),
			nn.ReLU(),
			nn.Linear(3072, emb_dim)
		)

		self.layer_norm = nn.LayerNorm(emb_dim)
		self.layer_norm2 = nn.LayerNorm(emb_dim)
	

	"""
	TransformerLayer.forward
		Runs a forward pass through the transformer
		layer's self attention (with residual),
		batch norm, feedfoward (with resudial),
		and batch norm.
	
	Args:
		x: torch.Tensor of size (B, N, D)
		is_causal: boolean causal masking flag
	
	Returns:
		torch.Tensor of size (B, N, D)
	"""
	def forward(self, x, is_causal):
		# run through residual attention layer (with causal mask if specified)
		# mask = torch.triu(torch.ones(x.shape[1], x.shape[1]), diagonal=1).bool().to(device)
		# x = x + self.attn_layer(x, x, x, is_causal=is_causal, attn_mask=mask)[0]
		x = x + self.attn_layer(x, is_causal)
		x = self.layer_norm(x)

		# run through feed forward network
		x = x + self.feed_forward(x)
		return self.layer_norm2(x)


"""
Transformer
	Full multiheaded and multilayered
	transformer.
"""
class Transformer(nn.Module):
	"""
	Transformer.__init__
		Configures interal list of
		transformer layers.
	
	Args:
		emb_dim: int embedding dimension
		num_heads: int number of heads
		num_layers: int number of layers
	"""
	def __init__(self, emb_dim, num_heads, num_layers):
		super().__init__()
		# build tranformer layers
		self.transformer_layers = nn.ModuleList(
			[TransformerLayer(emb_dim, num_heads) for _ in range(num_layers)]
		)
		
	
	"""
	Transformer.forward
		Runs a forward pass through the
		transformer.
	
	Args:
		x: torch.Tensor of size (B, N, D)
		is_causal: boolean causal masking flag
	"""
	def forward(self, x, is_causal):
		# run through layers
		for layer in self.transformer_layers:
			x = layer(x, is_causal)
		return x


class LoRALayer(torch.nn.Module):
	def __init__(self, in_dim, out_dim, rank, alpha):
		super().__init__()
		std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
		self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
		self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
		self.alpha = alpha

	def forward(self, x):
		x = self.alpha * (x @ self.A @ self.B)
		return x


class LinearLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)


"""
add_lora
    Replaces q_linear, k_linear, v_linear, and concat_linear in every
    MultiheadAttention layer with LinearLoRA wrappers. Call this after
    loading a pretrained checkpoint and before fine-tuning.

Args:
    model: GPT instance with pretrained weights already loaded
    rank:  int  LoRA rank (number of low-rank dimensions)
    alpha: float LoRA scaling factor (convention: alpha = 2 * rank)
"""
def add_lora(model, rank=8, alpha=16, device=device):
    for layer in model.transformer.transformer_layers:
        attn = layer.attn_layer.to(device)
        attn.q_linear      = LinearLoRA(attn.q_linear,      rank, alpha).to(device)
        attn.k_linear      = LinearLoRA(attn.k_linear,      rank, alpha).to(device)
        attn.v_linear      = LinearLoRA(attn.v_linear,      rank, alpha).to(device)
        attn.concat_linear = LinearLoRA(attn.concat_linear, rank, alpha).to(device)


"""
freeze_base_model
    Freezes every parameter except LoRA A/B matrices so that only the
    low-rank adapters are updated during fine-tuning.

Args:
    model: GPT instance with LoRA layers already injected via add_lora
"""
def freeze_base_model(model):
    for name, param in model.named_parameters():
        param.requires_grad = ".lora." in name or "classifier" in name or "token_embedding" in name