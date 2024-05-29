# VISION TRANSFORMERS MODULES AND MODEL DEFINITIONS
"""
Module definition for a Vision Transformer (ViT) as decribed in An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale by Dosovitskiy et al (2020)

Code is based on the publically available code for Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet by Yuan et al (2021)
Sourced from https://github.com/yitu-opensource/T2T-ViT

Regarding timm.layers.trunc_normal_(), see
https://github.com/huggingface/pytorch-image-models/blob/95ba90157fbbee293e8d10ac108ced2d9b990cbc/timm/layers/weight_init.py
"""

####################################
## Packages
####################################
import math
import typing
import numpy as np
import torch
import torch.nn as nn
import timm.layers 

####################################
## Custom Datatype for Type Hints
####################################
NoneFloat = typing.Union[None, float]

####################################
## Count Tokens
####################################
def count_tokens(w, h, k, s, p):
	""" Function to count how many tokens are produced from a given soft split

		Args:
			w (int): starting width
			h (int): starting height
			k (int): kernel size
			s (int): stride size
			p (int): padding size

		Returns:
			new_w (int): number of tokens along the width
			new_h (int): number of tokens along the height
			total (int): total number of tokens created

		See Also: 
		Formula taken from 
		https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold
		Assuming a 2D input, dilation = 1, and symmetric padding, kernel, and stride
	"""

	new_w = int(math.floor(((w + 2*p - (k-1) -1)/s)+1))
	new_h = int(math.floor(((h + 2*p - (k-1) -1)/s)+1))
	total = new_w * new_h

	return new_w, new_h, total

####################################
## Attention Module
####################################
class Attention(nn.Module):
	def __init__(self, 
				dim: int,
				chan: int,
				num_heads: int=1,
				qkv_bias: bool=False,
				qk_scale: NoneFloat=None):

		""" Attention Module

			Args:
				dim (int): input size of a single token
				chan (int): resulting size of a single token (channels)
				num_heads(int): number of attention heads in MSA
				qkv_bias (bool): determines if the qkv layer learns an addative bias
				qk_scale (NoneFloat): value to scale the queries and keys by; 
									if None, queries and keys are scaled by ``head_dim ** -0.5``
		"""

		super().__init__()

		## Define Constants
		self.num_heads = num_heads
		self.chan = chan
		self.head_dim = self.chan // self.num_heads
		if qk_scale == 'None' or qk_scale == None:
			self.scale = self.head_dim ** -0.5
			self.scale = float(self.scale)
		else:
			self.scale = float(qk_scale)
		assert self.chan % self.num_heads == 0, '"Chan" must be evenly divisible by "num_heads".'

		## Define Layers
		self.qkv = nn.Linear(dim, chan * 3, bias=qkv_bias)
		#### Each token gets projected from starting length (dim) to channel length (chan) 3 times (for each Q, K, V)
		self.proj = nn.Linear(chan, chan)

	def forward(self, x):
		B, N, C = x.shape
		## Dimensions: (batch, num_tokens, token_len)

		## Calcuate QKVs
		qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
		#### Dimensions: (3, batch, heads, num_tokens, chan/num_heads = head_dim)
		q, k, v = qkv[0], qkv[1], qkv[2]

		## Calculate Attention
		attn = torch.mul(q, self.scale) @ k.transpose(-2, -1)
		attn = attn.softmax(dim=-1)
		#### Dimensions: (batch, heads, num_tokens, num_tokens)

		## Attention Layer
		x = (attn @ v).transpose(1, 2).reshape(B, N, self.chan)
		#### Dimensions: (batch, heads, num_tokens, chan)

		## Projection Layers
		x = self.proj(x)

		## Skip Connection Layer
		v = v.transpose(1, 2).reshape(B, N, self.chan)
		x = v + x     
		#### Because the original x has different size with current x, use v to do skip connection

		return x

####################################
## Position Embedding
####################################
def get_sinusoid_encoding(num_tokens, token_len):
	""" Make Sinusoid Encoding Table

		Args:
			num_tokens (int): number of tokens
			token_len (int): length of a token
			
		Returns:
			(torch.FloatTensor) sinusoidal position encoding table
	"""

	def get_position_angle_vec(i):
		return [i / np.power(10000, 2 * (j // 2) / token_len) for j in range(token_len)]

	sinusoid_table = np.array([get_position_angle_vec(i) for i in range(num_tokens)])
	sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
	sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) 

	return torch.FloatTensor(sinusoid_table).unsqueeze(0)

####################################
## Patch Tokenization
####################################
class Patch_Tokenization(nn.Module):
	def __init__(self,
				img_size: tuple[int, int, int]=(1, 60, 100),
				patch_size: int=50,
				token_len: int=768):

		""" Patch Tokenization Module
			Args:
				img_size (tuple[int, int, int]): size of input (channels, height, width)
				patch_size (int): the side length of a square patch
				token_len (int): desired length of an output token
		"""
		super().__init__()
		
		## Defining Parameters
		self.img_size = img_size
		C, H, W = self.img_size
		self.patch_size = patch_size
		self.token_len = token_len
		assert H % self.patch_size == 0, 'Height of image must be evenly divisible by patch size.'
		assert W % self.patch_size == 0, 'Width of image must be evenly divisible by patch size.'
		self.num_tokens = (H / self.patch_size) * (W / self.patch_size)

		## Defining Layers
		self.split = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size, padding=0)
		self.project = nn.Linear((self.patch_size**2)*C, token_len)

	def forward(self, x):
		x = self.split(x).transpose(2,1)
		x = self.project(x)
		return x

####################################
## Neural Network Module
####################################
class NeuralNet(nn.Module):
	def __init__(self,
				in_chan: int,
				hidden_chan: NoneFloat=None,
				out_chan: NoneFloat=None,
				act_layer = nn.GELU):
		""" Neural Network Module

			Args:
				in_chan (int): number of channels (features) at input
				hidden_chan (NoneFloat): number of channels (features) in the hidden layer;
										if None, number of channels in hidden layer is the same as the number of input channels
				out_chan (NoneFloat): number of channels (features) at output;
										if None, number of output channels is same as the number of input channels
				act_layer(nn.modules.activation): torch neural network layer class to use as activation
		"""

		super().__init__()

		## Define Number of Channels
		hidden_chan = hidden_chan or in_chan
		out_chan = out_chan or in_chan

		## Make Parameters Integers
		assert isinstance(hidden_chan, int) or hidden_chan.is_integer(), "Hidden channels in Neural Network module must be an integer"
		in_chan = int(in_chan)
		hidden_chan = int(hidden_chan)
		out_chan = int(out_chan)

		## Define Layers
		self.fc1 = nn.Linear(in_chan, hidden_chan)
		self.act = act_layer()
		self.fc2 = nn.Linear(hidden_chan, out_chan)

	def forward(self, x):
		x = self.fc1(x)
		x = self.act(x)
		x = self.fc2(x)
		return x

####################################
## Encoding Block
####################################
class Encoding(nn.Module):

	def __init__(self,
	   dim: int,
	   num_heads: int=1,
	   hidden_chan_mul: float=4.,
	   qkv_bias: bool=False,
	   qk_scale: NoneFloat=None,
	   act_layer=nn.GELU, 
	   norm_layer=nn.LayerNorm):
		
		""" Encoding Block

			Args:
				dim (int): size of a single token
				num_heads(int): number of attention heads in MSA
				hidden_chan_mul (float): multiplier to determine the number of hidden channels (features) in the NeuralNet component
				qkv_bias (bool): determines if the qkv layer learns an addative bias
				qk_scale (NoneFloat): value to scale the queries and keys by; 
									if None, queries and keys are scaled by ``head_dim ** -0.5``
				act_layer(nn.modules.activation): torch neural network layer class to use as activation
				norm_layer(nn.modules.normalization): torch neural network layer class to use as normalization
		"""

		super().__init__()

		## Define Layers
		self.norm1 = norm_layer(dim)
		self.attn = Attention(dim=dim,
								chan=dim,
								num_heads=num_heads,
								qkv_bias=qkv_bias,
								qk_scale=qk_scale)
		self.norm2 = norm_layer(dim)
		self.neuralnet = NeuralNet(in_chan=dim,
									hidden_chan=int(dim*hidden_chan_mul),
									out_chan=dim,
									act_layer=act_layer)

	def forward(self, x):
		x = x + self.attn(self.norm1(x))
		x = x + self.neuralnet(self.norm2(x))
		return x

####################################
## ViT Backbone
####################################
class ViT_Backbone(nn.Module):
	def __init__(self,
				preds: int=1,
				token_len: int=768,
				num_tokens: int=500,
				num_heads: int=1,
				Encoding_hidden_chan_mul: float=4.,
				depth: int=12,
				qkv_bias=False,
				qk_scale=None,
				act_layer=nn.GELU,
				norm_layer=nn.LayerNorm):

		""" VisTransformer Backbone
			Args:
				preds (int): number of predictions to output
				token_len (int): length of a token
				num_tokens (int): number of tokens passed to the module
				num_heads(int): number of attention heads in MSA
				Encoding_hidden_chan_mul (float): multiplier to determine the number of hidden channels (features) in the NeuralNet component of the Encoding Module
				depth (int): number of encoding blocks in the model
				qkv_bias (bool): determines if the qkv layer learns an addative bias
				qk_scale (NoneFloat): value to scale the queries and keys by; 
				 if None, queries and keys are scaled by ``head_dim ** -0.5``
				act_layer(nn.modules.activation): torch neural network layer class to use as activation
				norm_layer(nn.modules.normalization): torch neural network layer class to use as normalization
		"""

		super().__init__()

		## Defining Parameters
		self.token_len = token_len
		self.num_tokens = num_tokens
		self.num_heads = num_heads
		self.Encoding_hidden_chan_mul = Encoding_hidden_chan_mul
		self.depth = depth

		## Defining Token Processing Components
		self.cls_token = nn.Parameter(torch.zeros(1, 1, self.token_len))
		self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(num_tokens=int(self.num_tokens+1), token_len=int(self.token_len)), requires_grad=False)

		## Defining Encoding blocks
		self.blocks = nn.ModuleList([Encoding(dim = self.token_len, 
											   num_heads = self.num_heads,
											   hidden_chan_mul = self.Encoding_hidden_chan_mul,
											   qkv_bias = qkv_bias,
											   qk_scale = qk_scale,
											   act_layer = act_layer,
											   norm_layer = norm_layer)
			 for i in range(self.depth)])

		## Defining Prediction Processing
		self.norm = norm_layer(self.token_len)
		self.head = nn.Linear(self.token_len, preds)

		## Make the class token sampled from a truncated normal distrobution 
		timm.layers.trunc_normal_(self.cls_token, std=.02)

	def forward(self, x):
		## Assumes x is already tokenized

		## Get Batch Size
		B = x.shape[0]
		## Concatenate Class Token
		x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
		## Add Positional Embedding
		x = x + self.pos_embed
		## Run Through Encoding Blocks
		for blk in self.blocks:
			x = blk(x)
		## Take Norm
		x = self.norm(x)
		## Make Prediction on Class Token
		x = self.head(x[:, 0])
		return x

####################################
## ViT Model
####################################
class ViT_Model(nn.Module):
	def __init__(self,
				img_size: tuple[int, int, int]=(1, 400, 100),
				patch_size: int=50,
				token_len: int=768,
				preds: int=1,
				num_heads: int=1,
				Encoding_hidden_chan_mul: float=4.,
				depth: int=12,
				qkv_bias=False,
				qk_scale=None,
				act_layer=nn.GELU,
				norm_layer=nn.LayerNorm):

		""" VisTransformer Model
			Args:
				img_size (tuple[int, int, int]): size of input (channels, height, width)
				patch_size (int): the side length of a square patch
				token_len (int): desired length of an output token
				preds (int): number of predictions to output
				num_heads(int): number of attention heads in MSA
				Encoding_hidden_chan_mul (float): multiplier to determine the number of hidden channels (features) in the NeuralNet component of the Encoding Module
				depth (int): number of encoding blocks in the model
				qkv_bias (bool): determines if the qkv layer learns an addative bias
				qk_scale (NoneFloat): value to scale the queries and keys by; 
									if None, queries and keys are scaled by ``head_dim ** -0.5``
				act_layer(nn.modules.activation): torch neural network layer class to use as activation
				norm_layer(nn.modules.normalization): torch neural network layer class to use as normalization
		"""
		super().__init__()

		## Defining Parameters
		self.img_size = img_size
		C, H, W = self.img_size
		self.patch_size = patch_size
		self.token_len = token_len
		self.num_heads = num_heads
		self.Encoding_hidden_chan_mul = Encoding_hidden_chan_mul
		self.depth = depth

		## Defining Patch Embedding Module
		self.patch_tokens = Patch_Tokenization(img_size,
												patch_size,
												token_len)
		num_tokens = self.patch_tokens.num_tokens

		## Defining ViT Backbone
		self.backbone = ViT_Backbone(preds,
									self.token_len,
									num_tokens,
									self.num_heads,
									self.Encoding_hidden_chan_mul,
									self.depth,
									qkv_bias,
									qk_scale,
									act_layer,
									norm_layer)
		## Initialize the Weights
		self.apply(self._init_weights)

	def _init_weights(self, m):
		""" Initialize the weights of the linear layers & the layernorms
		"""
		## For Linear Layers
		if isinstance(m, nn.Linear):
			## Weights are initialized from a truncated normal distrobution
			timm.layers.trunc_normal_(m.weight, std=.02)
			if isinstance(m, nn.Linear) and m.bias is not None:
				## If bias is present, bias is initialized at zero
				nn.init.constant_(m.bias, 0)
		## For Layernorm Layers
		elif isinstance(m, nn.LayerNorm):
			## Weights are initialized at one
			nn.init.constant_(m.weight, 1.0)
			## Bias is initialized at zero
			nn.init.constant_(m.bias, 0)

	@torch.jit.ignore ##Tell pytorch to not compile as TorchScript
	def no_weight_decay(self):
		""" Used in Optimizer to ignore weight decay in the class token
		"""
		return {'cls_token'}

	def forward(self, x):
		x = self.patch_tokens(x)
		x = self.backbone(x)
		return x

####################################
## Token Transformer
####################################
class TokenTransformer(nn.Module):

	def __init__(self,
				dim: int,
				chan: int,
				num_heads: int,
				hidden_chan_mul: float=1.,
				qkv_bias: bool=False,
				qk_scale: NoneFloat=None,
				act_layer=nn.GELU,
				norm_layer=nn.LayerNorm):

		""" Token Transformer Module

			Args:
				dim (int): size of a single token
				chan (int): resulting size of a single token 
				num_heads (int): number of attention heads in MSA 
				hidden_chan_mul (float): multiplier to determine the number of hidden channels (features) in the NeuralNet module
				qkv_bias (bool): determines if the attention qkv layer learns an addative bias
				qk_scale (NoneFloat): value to scale the queries and keys by; 
				if None, queries and keys are scaled by ``head_dim ** -0.5``
				act_layer(nn.modules.activation): torch neural network layer class to use as activation in the NeuralNet module
				norm_layer(nn.modules.normalization): torch neural network layer class to use as normalization
		"""

		super().__init__()

		## Define Layers
		self.norm1 = norm_layer(dim)
		self.attn = Attention(dim,
								chan=chan,
								num_heads=num_heads,
								qkv_bias=qkv_bias,
								qk_scale=qk_scale)
		self.norm2 = norm_layer(chan)
		self.neuralnet = NeuralNet(in_chan=chan,
									hidden_chan=int(chan*hidden_chan_mul),
									out_chan=chan,
									act_layer=act_layer)

	def forward(self, x):
		x = self.attn(self.norm1(x))
		x = x + self.neuralnet(self.norm2(x))
		return x

####################################
## Tokens-to-Token (T2T) Module
####################################
class Tokens2Token(nn.Module):
	def __init__(self, 
				img_size: tuple[int, int, int]=(1, 1700, 500), 
				softsplit_kernels: tuple[int, int, int]=(31, 3, 3),
				num_heads: int=1,
				token_chan:  int=64,
				token_len: int=768,
				hidden_chan_mul: float=1.,
				qkv_bias: bool=False,
				qk_scale: NoneFloat=None,
				act_layer=nn.GELU,
				norm_layer=nn.LayerNorm):

		""" Tokens-to-Token Module

			Args:
				img_size (tuple[int, int, int]): size of input (channels, height, width)
				softsplit_kernels (tuple[int int, int]): size of the square kernel for each of the soft split layers, sequentially
				num_heads (int): number of attention heads in MSA
				token_chan (int): number of token channels inside the TokenTransformers
				token_len (int): desired length of an output token
				hidden_chan_mul (float): multiplier to determine the number of hidden channels (features) in the NeuralNet module
				qkv_bias (bool): determines if the QKV layer in the Attention Module learns an addative bias
				qk_scale (NoneFloat): value to scale the Attention module queries and keys by; 
									if None, queries and keys are scaled by ``head_dim ** -0.5``
				act_layer(nn.modules.activation): torch neural network layer class to use as activation in the NeuralNet module
				norm_layer(nn.modules.normalization): torch neural network layer class to use as normalization
		"""

		super().__init__()

		## Seperating Image Size
		C, H, W = img_size
		self.token_chan = token_chan
		## Dimensions: (channels, height, width)

		## Define the Soft Split Layers
		self.k0, self.k1, self.k2 = softsplit_kernels
		self.s0, self.s1, self.s2 = [int((k+1)/2) for k in softsplit_kernels]
		self.p0, self.p1, self.p2 = [int((k+1)/4) for k in softsplit_kernels]
		self.soft_split0 = nn.Unfold(kernel_size=(self.k0, self.k0), stride=(self.s0, self.s0), padding=(self.p0, self.p0))
		self.soft_split1 = nn.Unfold(kernel_size=(self.k1, self.k1), stride=(self.s1, self.s1), padding=(self.p1, self.p1))
		self.soft_split2 = nn.Unfold(kernel_size=(self.k2, self.k2), stride=(self.s2, self.s2), padding=(self.p2, self.p2))

		## Determining Number of Output Tokens
		W, H, _ = count_tokens(w=W, h=H, k=self.k0, s=self.s0, p=self.p0)
		W, H, _ = count_tokens(w=W, h=H, k=self.k1, s=self.s1, p=self.p1)
		_, _, T = count_tokens(w=W, h=H, k=self.k2, s=self.s2, p=self.p2)
		self.num_tokens = T


		## Define the Transformer Layers
		self.transformer1 = TokenTransformer(dim =  C * self.k0 * self.k0,
											chan = token_chan,
											num_heads = num_heads,
											hidden_chan_mul = hidden_chan_mul,
											qkv_bias = qkv_bias,
											qk_scale = qk_scale,
											act_layer = act_layer,
											norm_layer = norm_layer)

		self.transformer2 = TokenTransformer(dim =  token_chan * self.k1 * self.k1,
											chan = token_chan,
											num_heads = num_heads,
											hidden_chan_mul = hidden_chan_mul,
											qkv_bias = qkv_bias,
											qk_scale = qk_scale,
											act_layer = act_layer,
											norm_layer = norm_layer)

		## Define the Projection Layer
		self.project = nn.Linear(token_chan * self.k2 * self.k2, token_len)

	def forward(self, x):

		B, C, H, W = x.shape
		### Dimensions: (batch, channels, height, width)

		## Initial Soft Split
		x = self.soft_split0(x).transpose(1, 2)

		## Token Transformer 1
		x = self.transformer1(x)

		## Reconstruct 2D Image
		W, H, _ = count_tokens(w=W, h=H, k=self.k0, s=self.s0, p=self.p0)
		x = x.transpose(1,2).reshape(B, self.token_chan, H, W)

		## Soft Split 1
		x = self.soft_split1(x).transpose(1, 2)

		## Token Transformer 2
		x = self.transformer2(x)

		## Reconstruct 2D Image
		W, H, _ = count_tokens(w=W, h=H, k=self.k1, s=self.s1, p=self.p1)
		x = x.transpose(1,2).reshape(B, self.token_chan, H, W)

		## Soft Split 2
		x = self.soft_split2(x).transpose(1, 2)

		## Project Tokens to desired length
		x = self.project(x)

		return x

####################################
## T2T-ViT Model
####################################
class T2T_ViT(nn.Module):
	def __init__(self, 
				img_size: tuple[int, int, int]=(1, 1700, 500),
				softsplit_kernels: tuple[int, int, int]=(31, 3, 3),
				preds: int=1,
				token_len: int=768,
				token_chan:  int=64,
				num_heads: int=1,
				T2T_hidden_chan_mul: float=1.,
				Encoding_hidden_chan_mul: float=4.,
				depth: int=12,
				qkv_bias=False,
				qk_scale=None,
				act_layer=nn.GELU,
				norm_layer=nn.LayerNorm):

		""" Tokens-to-Token VisTransformer Model

			Args:
				img_size (tuple[int, int, int]): size of input (channels, height, width)
				softsplit_kernels (tuple[int int, int]): size of the square kernel for each of the soft split layers, sequentially
				preds (int): number of predictions to output
				token_len (int): desired length of an output token
				token_chan (int): number of token channels inside the TokenTransformers
				num_heads(int): number of attention heads in MSA (only works if =1)
				T2T_hidden_chan_mul (float): multiplier to determine the number of hidden channels (features) in the NeuralNet component of the Tokens-to-Token (T2T) Module
				Encoding_hidden_chan_mul (float): multiplier to determine the number of hidden channels (features) in the NeuralNet component of the Encoding Module
				depth (int): number of encoding blocks in the model
				qkv_bias (bool): determines if the qkv layer learns an addative bias
				qk_scale (NoneFloat): value to scale the queries and keys by; 
									if None, queries and keys are scaled by ``head_dim ** -0.5``
				act_layer(nn.modules.activation): torch neural network layer class to use as activation
				norm_layer(nn.modules.normalization): torch neural network layer class to use as normalization
		"""

		super().__init__()

		## Defining Parameters
		self.img_size = img_size
		C, H, W = self.img_size
		self.softsplit_kernels = softsplit_kernels
		self.token_len = token_len
		self.token_chan = token_chan
		self.num_heads = num_heads
		self.T2T_hidden_chan_mul = T2T_hidden_chan_mul
		self.Encoding_hidden_chan_mul = Encoding_hidden_chan_mul
		self.depth = depth

		## Defining Tokens-to-Token Module
		self.tokens_to_token = Tokens2Token(img_size = self.img_size, 
											softsplit_kernels = self.softsplit_kernels,
											num_heads = self.num_heads,
											token_chan = self.token_chan,
											token_len = self.token_len,
											hidden_chan_mul = self.T2T_hidden_chan_mul,
											qkv_bias = qkv_bias,
											qk_scale = qk_scale,
											act_layer = act_layer,
											norm_layer = norm_layer)
		self.num_tokens = self.tokens_to_token.num_tokens

		## Defining Token Processing Components
		self.vit_backbone = ViT_Backbone(preds = preds,
										num_tokens = self.num_tokens,
										token_len = self.token_len,
										num_heads = self.num_heads,
										Encoding_hidden_chan_mul = self.Encoding_hidden_chan_mul,
										depth = self.depth,
										qkv_bias = qkv_bias,
										qk_scale = qk_scale,
										act_layer = act_layer,
										norm_layer = norm_layer)

		## Initialize the Weights
		self.apply(self._init_weights)

	def _init_weights(self, m):
		""" Initialize the weights of the linear layers & the layernorms
		"""
		## For Linear Layers
		if isinstance(m, nn.Linear):
			## Weights are initialized from a truncated normal distrobution
			timm.layers.trunc_normal_(m.weight, std=.02)
			if isinstance(m, nn.Linear) and m.bias is not None:
				## If bias is present, bias is initialized at zero
				nn.init.constant_(m.bias, 0)
		## For Layernorm Layers
		elif isinstance(m, nn.LayerNorm):
			## Weights are initialized at one
			nn.init.constant_(m.weight, 1.0)
			## Bias is initialized at zero
			nn.init.constant_(m.bias, 0)
			
	@torch.jit.ignore ##Tell pytorch to not compile as TorchScript
	def no_weight_decay(self):
		""" Used in Optimizer to ignore weight decay in the class token
		"""
		return {'cls_token'}

	def forward(self, x):
		x = self.tokens_to_token(x)
		x = self.vit_backbone(x)
		return x