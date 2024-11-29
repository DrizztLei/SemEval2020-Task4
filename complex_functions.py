import torch
import torch.nn as nn
import numpy as np


def Complexbmm(x, y):
	''' Batch Matrix Multiplication for Complex Numbers '''
	real = torch.matmul(x.real, y.real) - torch.matmul(x.imag, y.imag)
	imag = torch.matmul(x.real, y.imag) + torch.matmul(x.imag, y.real)
	out  = torch.view_as_complex(torch.stack([real, imag], -1))
	return out

class ComplexLinear(nn.Module):
	''' [nn.Linear] Fully Connected Layer for Complex Numbers '''
	def __init__(self, in_features, out_features, bias=True):
		super(ComplexLinear, self).__init__()
		self.real_linear = nn.Linear(in_features, out_features, bias=bias)
		self.imag_linear = nn.Linear(in_features, out_features, bias=bias)
	
	def forward(self, x):
		real = self.real_linear(x.real) - self.imag_linear(x.imag)
		imag = self.real_linear(x.imag) + self.imag_linear(x.real)
		out  = torch.view_as_complex(torch.stack([real, imag], -1))
		return out


class ComplexDropout(nn.Module):
	''' [nn.Dropout] DropOut for Complex Numbers '''
	def __init__(self, p=0.1, inplace=False):
		super(ComplexDropout, self).__init__()
		self.drop = nn.Dropout(p=p, inplace=inplace)
	
	def forward(self, x):
		x.imag = x.imag + 1e-10
		mag, phase = self.drop(x.abs()), x.angle()
		real, imag = mag * torch.cos(phase), mag * torch.sin(phase)
		out  = torch.view_as_complex(torch.stack([real, imag], -1))
		return out

class ComplexSoftMax(nn.Module):
	''' [nn.Softmax] SoftMax for Complex Numbers '''
	def __init__(self, dim=-1):
		super(ComplexSoftMax, self).__init__()
		self.softmax = nn.Softmax(dim=dim)
	
	def forward(self, x):
		x.imag = x.imag + 1e-10
		mag, phase = self.softmax(x.abs()), x.angle()
		real, imag = mag * torch.cos(phase), mag * torch.sin(phase)
		out  = torch.view_as_complex(torch.stack([real, imag], -1))
		return out

class ComplexLayerNorm(nn.Module):
	''' [nn.LayerNorm] LayerNorm for Complex Numbers '''
	def __init__(self, normal_shape, affine=True, epsilon=1e-10):
		super(ComplexLayerNorm, self).__init__()
		if isinstance(normal_shape, int):
			normal_shape = (normal_shape,)
		else:
			normal_shape = (normal_shape[-1],)
		self.normal_shape = torch.Size(normal_shape)
		self.epsilon = epsilon
		self.affine  = affine
		if self.affine:
			self.gamma = nn.Parameter(torch.Tensor(*normal_shape))
			self.beta  = nn.Parameter(torch.Tensor(*normal_shape))
		else:
			self.register_parameter('gamma', None)
			self.register_parameter('beta', None)
		self.reset_parameters()
	
	def reset_parameters(self):
		if self.affine:
			self.gamma.data.fill_(1)
			self.beta.data.zero_()
	
	def forward(self, x):
		dim  = list(range(1,len(x.shape)))
		mean = torch.view_as_complex(torch.stack((x.real.mean(dim=dim, keepdim=True), x.imag.mean(dim=dim, keepdim=True)),-1))
		x_mean = (x - mean)
		std  = ((x_mean * x_mean.conj()).abs() + self.epsilon).sqrt()
		y    = torch.view_as_complex(torch.stack((x_mean.real/std, x_mean.imag/std),-1))
		if self.affine:
			y = (self.gamma * y) + self.beta
		return y


class ComplexCrossEntropyLoss(nn.Module):
	
	def __init__(self, weight=None, size_average=None, ignore_index: int = -100,
	             reduce=None, reduction: str = 'mean') -> None:
		super(ComplexCrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
		self.ignore_index = ignore_index
		
	
	def forward(self, inputs, targets):
		
		if torch.is_complex(inputs):
			real_loss = nn.CrossEntropyLoss(inputs.real, targets)
			imag_loss = nn.CrossEntropyLoss(inputs.imag, targets)
			return (real_loss + imag_loss) / 2
		else:
			return nn.CrossEntropyLoss(inputs, targets)


def param(nnet, Mb=True):
	return np.round(sum([param.nelement() for param in nnet.parameters()]) / 10 ** 6 if Mb else neles, 2)


class ComplexScaledDotProductAttention(nn.Module):
	''' Scaled Dot-Product for Complex Numbers '''
	
	def __init__(self, temperature, attn_dropout=0.1):
		super().__init__()
		self.temperature = temperature
		self.dropout = nn.Dropout(attn_dropout)
		self.softmax = nn.Softmax(dim=-1)
	
	def forward(self, q, k, v, mask=None):
		attn = Complexbmm(q / self.temperature, k.conj().transpose(-1, -2)).abs()
		if mask is not None:
			attn = attn.masked_fill(mask == 0, -1e9)
		attn = self.dropout(self.softmax(attn))
		output = torch.view_as_complex(torch.stack([torch.matmul(attn, v.real), torch.matmul(attn, v.imag)], -1))
		# output   = Complexbmm(attn, v)
		return output, attn


class ComplexMultiHeadAttention(nn.Module):
	''' Multi-Head Attention module for Complex Numbers '''
	
	def __init__(self, n_head, f_in, dropout=0.1):
		super().__init__()
		
		self.n_head = n_head
		self.d_k = f_in // n_head
		self.d_v = f_in // n_head
		
		self.w_qs = ComplexLinear(f_in, n_head * self.d_k, bias=False)
		self.w_ks = ComplexLinear(f_in, n_head * self.d_k, bias=False)
		self.w_vs = ComplexLinear(f_in, n_head * self.d_v, bias=False)
		self.fc = ComplexLinear(n_head * self.d_v, f_in, bias=False)
		
		self.attention = ComplexScaledDotProductAttention(temperature=self.d_k ** 0.5)
		self.dropout = ComplexDropout(dropout)
		self.layer_norm = ComplexLayerNorm(f_in, epsilon=1e-6)
	
	def forward(self, q, k, v, mask=None):
		d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
		sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
		
		residual = q
		
		# Pass through the pre-attention projection: b x lq x (n*dv)
		# Separate different heads: b x lq x n x dv
		q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
		k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
		v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
		
		# Transpose for attention dot product: b x n x lq x dv
		q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
		
		if mask is not None:
			mask = mask.unsqueeze(1)  # For head axis broadcasting.
		
		q, attn = self.attention(q, k, v, mask=mask)
		
		# Transpose to move the head dimension back: b x lq x n x dv
		# Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
		q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
		q = self.dropout(self.fc(q))
		q += residual
		
		q = self.layer_norm(q)
		
		return q, attn


class ComplexPositionwiseFeedForward(nn.Module):
	''' A two-feed-forward-layer module for Complex Numbers '''
	
	def __init__(self, d_in, d_hid, dropout=0.1):
		super().__init__()
		self.w_1 = ComplexLinear(d_in, d_hid)  # position-wise
		self.w_2 = ComplexLinear(d_hid, d_in)  # position-wise
		self.l_norm = ComplexLayerNorm(d_in, epsilon=1e-6)
		self.dropout = ComplexDropout(dropout)
	
	def forward(self, x):
		residual = x
		
		x = self.w_2(self.w_1(x))
		x = self.dropout(x)
		x += residual
		
		x = self.l_norm(x)
		
		return x


class ComplexSelfAttention(nn.Module):
	''' Input: [Batch x Time x Features]  Complex Self-Attention for Complex Numbers '''
	
	def __init__(self, f_in, f_out, n_head, dropout=0.1):
		super(ComplexSelfAttention, self).__init__()
		self.slf_attn = ComplexMultiHeadAttention(n_head, f_in, dropout=dropout)
		self.pos_ffn = ComplexPositionwiseFeedForward(f_in, f_out, dropout=dropout)
	
	def forward(self, enc_input, slf_attn_mask=None):
		enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
		enc_output = self.pos_ffn(enc_output)
		return enc_output, enc_slf_attn
