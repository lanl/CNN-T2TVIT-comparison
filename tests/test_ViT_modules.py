# TESTING FOR TOKENS-TO-TOKEN VISTRANFORMER (SKYLAR'S VERSION) MODEL
"""
"""

#############################
## Packages
#############################
import pytest
import sys
import os
import torch
import torch.nn

sys.path.insert(0, os.path.abspath('./modules'))
import ViT_modules as model

#############################
## Define Fixtures
#############################
@pytest.fixture
def token_len():
	return 7*7

@pytest.fixture
def chan():
	return 64

@pytest.fixture
def num_tokens():
	return 50

@pytest.fixture
def num_heads():
	return 4

@pytest.fixture
def batchsize():
	return 8

@pytest.fixture
def H():
	return 100

@pytest.fixture
def W():
	return 300

@pytest.fixture
def P():
	return 25

@pytest.fixture
def proj():
	return 8

#############################
## Test Count Tokens
#############################
class Test_CountTokens():
	""" """
	def test_count(self):
		""" Tests that the count tokens function accuractly counts the tokens for a small image """
		(W, H, T) = model.count_tokens(w=5, h=8, k=3, s=2, p=1)
		assert (W, H, T) == (3, 4, 12)


#############################
## Test Attention Module
#############################
class Test_Attention():
	""" """
	def test_shape(self, token_len, chan, num_tokens, num_heads, batchsize):
		""" Tests that the Attention Module outputs the correct shape """
		inpt = torch.rand(batchsize, num_tokens, token_len)
		A = model.Attention(dim=token_len,
							chan = chan,
							num_heads= num_heads,
							qkv_bias = False,
							qk_scale = None)
		A.eval()
		outpt = A.forward(inpt)
		assert outpt.shape == torch.Size([batchsize, num_tokens, chan])

	def test_qkscale_None(self, token_len, chan, num_tokens, num_heads, batchsize):
		""" Tests that the Attention Module correctly assigns the scale value when qk_scale=None """
		inpt = torch.rand(batchsize, num_tokens, token_len)
		A = model.Attention(dim=token_len,
							chan = chan,
							num_heads= num_heads,
							qkv_bias = False,
							qk_scale = None)
		A.eval()
		correct_scale = (chan // num_heads) ** -0.5
		assert A.scale == correct_scale

	def test_qkscale_value(self, token_len, chan, num_tokens, num_heads, batchsize):
		""" Tests that the Attention Module correctly assigns the scale value when qk_scale=value """
		inpt = torch.rand(batchsize, num_tokens, token_len)
		correct_scale = 1.5
		A = model.Attention(dim=token_len,
							chan = chan,
							num_heads= num_heads,
							qkv_bias = False,
							qk_scale = correct_scale)
		A.eval()
		assert A.scale == correct_scale

	def test_invalid(self, token_len, chan, num_tokens, num_heads, batchsize):
		""" Tests that the Attention Module errors out when the channel and number of heads are not divisible """
		with pytest.raises(AssertionError):
			inpt = torch.rand(batchsize, num_tokens, token_len)
			A = model.Attention(dim=token_len,
								chan = 65,
								num_heads= num_heads,
								qkv_bias = False,
								qk_scale = None)
			A.eval()
			outpt = A.forward(inpt)


#############################
## Test Sinusoidal Positional Encoding
#############################
class Test_PositionEncoding():
	""" """
	def test_shape(self, token_len, num_tokens):
		""" Tests that the positional encoding outputs the correct shape """
		pos_enc = model.get_sinusoid_encoding(num_tokens, token_len)
		assert pos_enc.shape == torch.Size([1, num_tokens, token_len])


#############################
## Test Patch Tokenization
#############################
class Test_Patch_Tokenization():
	""" """
	def test_numtokens(self, token_len, H, W, P):
		""" Tests that the Patch Tokenization Modudle correctly calculates the number of tokens """
		num_tokens = (H/P) * (W/P)
		PT = model.Patch_Tokenization(img_size = (1, H, W),
										patch_size = P,
										token_len = token_len)
		assert PT.num_tokens == num_tokens

	def test_splitshape(self, token_len, H, W, P, batchsize):
		""" Tests that the Patch Tokenization module correctly splits an input image into tokens """
		num_tokens = int((H/P) * (W/P))
		inpt = torch.rand(batchsize, 1, H, W)
		PT = model.Patch_Tokenization(img_size = (1, H, W),
										patch_size = P,
										token_len = token_len)
		outpt = PT.split(inpt)
		assert outpt.shape == torch.Size([batchsize, P**2, num_tokens])

	def test_finalshape(self, token_len, H, W, batchsize, P):
		""" Tests that the Patch Tokenization module correctly splits an input image into tokens """
		num_tokens = int((H/P) * (W/P))
		inpt = torch.rand(batchsize, 1, H, W)
		PT = model.Patch_Tokenization(img_size = (1, H, W),
										patch_size = P,
										token_len = token_len)
		PT.eval()
		outpt = PT.forward(inpt)
		assert outpt.shape == torch.Size([batchsize, num_tokens, token_len])

	def test_invalidH(self, token_len, H, W, P):
		""" Tests that the Patch Tokenization Module error when the image height is not divisible by the patch size """
		with pytest.raises(AssertionError):
			PT = model.Patch_Tokenization(img_size = (1, H+5, W),
											patch_size = P,
											token_len = token_len)
	def test_invalidW(self, token_len, H, W, P):
		""" Tests that the Patch Tokenization Module error when the image width is not divisible by the patch size """
		with pytest.raises(AssertionError):
			PT = model.Patch_Tokenization(img_size = (1, H, W+5),
											patch_size = P,
											token_len = token_len)


#############################
## Test Neural Network Module
#############################
class Test_NeuralNet():
	""" """
	def test_shape(self, token_len, chan, num_tokens, num_heads, batchsize):
		""" Tests that the Neural Network Module outputs the correct shape """
		inpt = torch.rand(batchsize, num_tokens, chan)
		NN = model.NeuralNet(in_chan = chan,
							hidden_chan = 1.5 * chan,
							out_chan = chan,
							act_layer = torch.nn.GELU)
		NN.eval()
		outpt = NN.forward(inpt)
		assert outpt.shape == torch.Size([batchsize, num_tokens, chan])

	def test_invalid(self, token_len, chan, num_tokens, num_heads, batchsize):
		""" Tests that the Neural Network Module errors out when the hidden channels are not an integer """
		with pytest.raises(AssertionError):
			inpt = torch.rand(batchsize, num_tokens, chan)
			NN = model.NeuralNet(in_chan = chan,
								hidden_chan = 1.3 * chan,
								out_chan = chan,
								act_layer = torch.nn.GELU)
			NN.eval()
			outpt = NN.forward(inpt)

#############################
## Test Encoding Module
#############################
class Test_Enoding():
	""" """
	def test_shape(self, token_len, chan, num_tokens, num_heads, batchsize):
		""" Tests that the Encoding Module outputs the correct shape """
		inpt = torch.rand(batchsize, num_tokens, chan)
		E = model.Encoding(dim = chan,
							num_heads = num_heads,
							hidden_chan_mul = 1.5,
							qkv_bias = False,
							qk_scale = None,
							act_layer = torch.nn.GELU, 
							norm_layer = torch.nn.LayerNorm)
		E.eval()
		outpt = E.forward(inpt)
		assert outpt.shape == torch.Size([batchsize, num_tokens, chan])


#############################
## Test ViT Backbone
#############################
class Test_ViT_Backbone():
	""" """
	def test_shape(self, chan, num_tokens, num_heads, batchsize, H, W):
		""" Tests that the Tokens-to-Token (T2T) VisTransformer (ViT) Module outputs the correct shape """
		inpt = torch.rand(batchsize, num_tokens, chan)
		preds = 1
		ViT_Backbone = model.ViT_Backbone(preds = preds,
											token_len = chan,
											num_tokens = num_tokens,
											num_heads = num_heads,
											Encoding_hidden_chan_mul = 1.5,
											depth = 4,
											qkv_bias=False,
											qk_scale=None,
											act_layer=torch.nn.GELU,
											norm_layer=torch.nn.LayerNorm)
		ViT_Backbone.eval()
		outpt = ViT_Backbone.forward(inpt)
		assert outpt.shape == torch.Size([batchsize, preds])


#############################
## Test ViT Model
#############################
class Test_ViT_Model():
	""" """
	def test_shape(self, chan, num_tokens, num_heads, batchsize, H, W, P):
		""" Tests that the Tokens-to-Token (T2T) VisTransformer (ViT) Module outputs the correct shape """
		inpt = torch.rand(batchsize, 1, H, W)
		preds = 1
		ViT = model.ViT_Model(img_size = (1, H, W),
								patch_size = P,
								token_len = chan,
								preds = preds,
								num_heads = num_heads,
								Encoding_hidden_chan_mul = 1.5,
								depth = 4,
								qkv_bias=False,
								qk_scale=None,
								act_layer=torch.nn.GELU,
								norm_layer=torch.nn.LayerNorm)
		ViT.eval()
		outpt = ViT.forward(inpt)
		assert outpt.shape == torch.Size([batchsize, preds])


#############################
## Test Token Transformer Module
#############################
class Test_TokenTransformer():
	""" """
	def test_shape(self, token_len, chan, num_tokens, num_heads, batchsize):
		""" Tests that the Token Transformer Module outputs the correct shape """
		inpt = torch.rand(batchsize, num_tokens, token_len)
		TT = model.TokenTransformer(dim = token_len,
					    			chan = chan,
					    			num_heads = num_heads,
					    			hidden_chan_mul = 1.5,
					    			qkv_bias = False,
					    			qk_scale = None,
					    			act_layer = torch.nn.GELU,
					    			norm_layer = torch.nn.LayerNorm)
		TT.eval()
		outpt = TT.forward(inpt)
		assert outpt.shape == torch.Size([batchsize, num_tokens, chan])


######################################
## Test Tokens-to-Token (T2T) Module
######################################
class Test_Tokens2Token():
	""" """
	def test_shape(self, token_len, chan, num_tokens, num_heads, batchsize, H, W, proj):
		""" Tests that the Tokens-to-Token (T2T) Module outputs the correct shape """
		inpt = torch.rand(batchsize, 1, H, W)
		T2T = model.Tokens2Token(img_size = (1, H, W),
								softsplit_kernels = (7, 3, 3),
								num_heads = num_heads,
								token_chan = chan,
								token_len = proj,
								hidden_chan_mul = 1.5,
								qkv_bias = False,
								qk_scale = None,
								act_layer = torch.nn.GELU,
								norm_layer = torch.nn.LayerNorm)
		T2T.eval()
		outpt = T2T.forward(inpt)

		# Calculate Output Shape
		w_new, h_new, _ = model.count_tokens(w=W, h=H, k=T2T.k0, s=T2T.s0, p=T2T.p0)
		w_new, h_new, _ = model.count_tokens(w=w_new, h=h_new, k=T2T.k1, s=T2T.s1, p=T2T.p1)
		_, _, T = model.count_tokens(w=w_new, h=h_new, k=T2T.k2, s=T2T.s2, p=T2T.p2)

		assert outpt.shape == torch.Size([batchsize, T, proj])

############################################################
## Test Tokens-to-Token (T2T) VisTransformer (ViT) Module
###########################################################
class Test_T2TViT():
	""" """
	def test_shape(self, token_len, chan, num_tokens, num_heads, batchsize, H, W, proj):
		""" Tests that the Tokens-to-Token (T2T) VisTransformer (ViT) Module outputs the correct shape """
		inpt = torch.rand(batchsize, 1, H, W)
		preds = 1
		T2TViT = model.T2T_ViT(img_size = (1, H, W),
								softsplit_kernels = (7, 3, 3),
								preds = preds, 
								token_len = proj,
								token_chan = chan,
								num_heads = num_heads,
								T2T_hidden_chan_mul = 1.5,
								Encoding_hidden_chan_mul = 1.5,
								depth = 4,
								qkv_bias = False,
								qk_scale = None,
								act_layer = torch.nn.GELU,
								norm_layer = torch.nn.LayerNorm)
		T2TViT.eval()
		outpt = T2TViT.forward(inpt)
		assert outpt.shape == torch.Size([batchsize, preds])