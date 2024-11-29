import pickle
import re
import torch
from bidict import bidict
from collections import defaultdict, OrderedDict
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer

from utils.text_to_uri import standardized_uri
import numpy as np
np.int = np.int_
from numba import njit
from scipy.sparse import coo_matrix, save_npz, load_npz
from scipy.optimize import fsolve
from random import choice
import threading
from scipy.linalg import expm
import networkx as nx

# %%

class ADVGraphUtils:
	__instance = None

	@classmethod
	def get_instance(cls):
		if cls.__instance is None:
			cls.__instance = cls()
		return cls.__instance

	def batch_init(self):
		if self.init_finished:
			print("graph has been inited before, skip it...")
			return
		# self.semaphore = threading.Semaphore()
		print('graph init...')
		# self.load_mp_all_by_pickle(self.args['mp_pickle_path'])
		self.init(is_load_necessary_data=True)
		print('merge graph by downgrade...')
		self.merge_graph_by_downgrade()
		print('reduce graph noise...')
		mp = self.reduce_graph_noise()  # 根据黑白名单，停用词，边权等信息进行简单修剪
		print('encode graph...')
		word, word_index, edge_index, edge_weight = self.encode_sorted_index(mp)
		print('compute adj matrix...')
		self.compute_adj_with_matrix(word, edge_index, edge_weight, max_order=2, matrix_path=None)
		print('compute adj matrix done!')

		print('compute the expm adj matrix')
		# self.ea = expm(self.adj_matrix)
		print('compute the expm adj matrix done!')

		self.init_finished = True

	def __init__(self):
		self.semaphore = threading.Semaphore()
		self.encode_semaphore = threading.Semaphore()
		# self.semaphore.acquire()
		self.graph_real_words_seq = None
		self.mp_all = defaultdict(set)  # 记录 id -> id 边的关系
		self.words_to_id = bidict()  # 将一个词映射为 id，仅在 encode_mp 时候构建
		self.words_encode_idx = 0  # 在 encode_mp 时构建
		self.local_graph_words_encode_idx = 0
		self.graph_words_to_id = bidict()
		self.local_graph_words_to_id = bidict()
		self.source_entities_limit = 0
		self.conceptnet_numberbatch_en = dict()
		# 这里之所以不用 GPT2/Roberta tokenizer 是因为空格会被分割为 Ġ
		self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

		self.args = {
			'n_gram': 3,
			'mp_pickle_path': './conceptnet5/res_all.pickle',
			'conceptnet_numberbatch_en_path': './conceptnet5/numberbatch-en.txt',
			'reduce_noise_args': {
				# 白名单相对优先
				'relation_white_list': [],
				# ['/r/RelatedTo', '/r/IsA', '/r/PartOf', '/r/HasA', '/r/UsedFor', '/r/CapableOf', '/r/AtLocation', '/r/Causes', '/r/HasProperty'],
				'relation_black_list': ['/r/ExternalURL', '/r/Synonym', '/r/Antonym',
										'/r/DistinctFrom', '/r/dbpedia/genre', '/r/dbpedia/influencedBy'],
				'stop_words': ['a', 'able', 'about', 'across', 'after', 'all', 'almost', 'also', 'am', 'among', 'an',
							   'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'but', 'by', 'can', 'cannot',
							   'could', 'dear', 'did', 'do', 'does', 'either', 'else', 'ever', 'every', 'for', 'from',
							   'get', 'got', 'had', 'has', 'have', 'he', 'her', 'hers', 'him', 'his', 'how', 'however',
							   'i', 'if', 'in', 'into', 'is', 'it', 'its', 'just', 'least', 'let', 'like', 'likely',
							   'may', 'me', 'might', 'most', 'must', 'my', 'neither', 'no', 'nor', 'not', 'of', 'off',
							   'often', 'on', 'only', 'or', 'other', 'our', 'own', 'rather', 'said', 'say', 'says',
							   'she', 'should', 'since', 'so', 'some', 'than', 'that', 'the', 'their', 'them', 'then',
							   'there', 'these', 'they', 'this', 'tis', 'to', 'too', 'twas', 'us', 'wants', 'was', 'we',
							   'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with',
							   'would', 'yet', 'you', 'your'],
				'weight_limit': 1.5,
				'edge_count_limit': 100,  # 保留权重最大的 edge_count_limit 条边
			}
		}
		self.adj_matrix = None
		self.order_matrix = None
		self.sp_order_matrix = None
		self.matrix_prefix = "matrix_"
		self.matrix_path = "matrix/"
		self.init_finished = False
		self.batch_init()
		# self.semaphore.release()

	def load_mp_all_by_pickle(self, fpath):
		"""
		从 pickle 中加载 conceptnet 图
		:param fpath:
		:return:
		"""
		graph_zip = None
		with open(fpath, 'rb') as f:
			graph_zip = pickle.load(f)
		self.mp_all, = graph_zip
		return graph_zip

	def get_mp(self):
		return self.mp_all

	def reduce_graph_noise(self, is_overwrite=True):
		"""
		基于 relation type 以及 edge weight 降低图的噪声
		:param is_overwrite: 是否写入到 self.mp 中
		:return:
		"""
		relation_white_list = self.args['reduce_noise_args']['relation_white_list']
		relation_black_list = self.args['reduce_noise_args']['relation_black_list']
		stop_words = self.args['reduce_noise_args']['stop_words']
		weight_limit = self.args['reduce_noise_args']['weight_limit']
		edge_count_limit = self.args['reduce_noise_args']['edge_count_limit']
		is_black_list = True  # 默认是开启黑名单的

		if len(relation_white_list) != 0:
			# 如果白名单里有则启用白名单
			is_black_list = False

		new_mp = defaultdict(set)  # 记录 id -> id 边的关系
		sorted_item = sorted(self.mp_all.items(), key=lambda x: x[0])
		for item in sorted_item:
			key = item[0]
			values = item[1]
			st_words = key
			if st_words in stop_words:
				# 停用词跳过
				continue

			# 取 values 按 edge_weight 从大到小排序以后的前 edge_count_limit 个（也可按概率选取）
			to_values = sorted(list(values), key=lambda x: x[2], reverse=True)
			edge_count = 0
			for value in to_values:
				to_words = value[0]
				to_relation = value[1]
				to_weight = value[2]
				if to_words in stop_words:
					# 停用词跳过
					continue
				if to_weight < weight_limit:
					# 边权较低的跳过
					continue
				if is_black_list:
					# 如果黑名单开启并且当前 relation 在黑名单里跳过
					if to_relation in relation_black_list:
						continue
				else:
					# 白名单下如果 relation 不在白名单里跳过
					if to_relation not in relation_white_list:
						continue
				new_mp[st_words].add((to_words, to_relation, to_weight))
				edge_count += 1
				if edge_count >= edge_count_limit:
					break

		if is_overwrite:
			self.mp_all = new_mp
		return new_mp

	def merge_graph_by_downgrade(self, is_overwrite=True):
		"""
		降级合并 mp 图，将形如 /c/en/apple/n 降级为 /c/en/apple，并省略 /c/en/
		:param is_overwrite: 是否将最终的结果直接写入 self 中的对象
		:return: 降级以后的 mp
		"""
		new_mp = defaultdict(set)  # 记录 id -> id 边的关系
		refine_sent = lambda s: re.match('/c/en/([^/]+)', s).group(1)
		sorted_item = sorted(self.mp_all.items(), key=lambda x: x[0])
		for item in sorted_item:
			key = item[0]
			values = item[1]
			st_words = refine_sent(key)
			for value in values:
				to_words = refine_sent(value[0])
				to_relation = value[1]
				to_weight = value[2]
				new_mp[st_words].add((to_words, to_relation, to_weight))
		if is_overwrite:
			self.mp_all = new_mp
		return new_mp

	def init(self, is_load_necessary_data=True):
		"""
		load 部分数据初始化
		:return:
		"""

		# self.__init__()
		if is_load_necessary_data:
			self.load_mp_all_by_pickle(self.args['mp_pickle_path'])
			self.load_conceptnet_numberbatch(self.args['conceptnet_numberbatch_en_path'])

	def get_features_from_words(self, words):
		"""
		获取 words 的词向量
		:param words:
		:return:
		"""
		words = standardized_uri('en', words).replace('/c/en/', '')
		res = self.conceptnet_numberbatch_en.get(words)
		if res is None:
			"""todo: 解决 OOV 问题，待定，暂时用 ## 代替"""
			# res = self.conceptnet_numberbatch_en.get('##')
			res = self.get_default_oov_feature()
		return res

	def get_default_oov_feature(self):
		# 默认的 oov feature
		return [0.0 for _ in range(300)]

	def load_conceptnet_numberbatch(self, fpath):
		"""
		从 numberbatch 中加载每个词的词向量
		:param fpath:
		:return:
		"""
		if len(self.conceptnet_numberbatch_en) != 0:
			# 如果已经加载则不用管了
			return
		self.conceptnet_numberbatch_en.clear()
		with open(fpath, 'r', encoding='UTF-8') as f:
			n, vec_size = list(map(int, f.readline().split(' ')))
			print('load conceptnet numberbatch: ', n, vec_size)
			for i in range(n):
				tmp_data = f.readline().split(' ')
				words = str(tmp_data[0])
				vector = list(map(float, tmp_data[1:]))
				self.conceptnet_numberbatch_en[words] = vector
			print('load conceptnet numberbatch done!')

	def get_words_from_id(self, id):
		"""
		从 id 获取对应的 words
		:param id:
		:return:
		"""
		return self.words_to_id.inverse.get(id)

	def get_local_words_from_id(self, id):

		return self.local_graph_words_to_id.inverse.get(id)

	def load_dense_matrix(self, matrix_path):
		import os

		# 设置需要列举文件的目录路径
		directory_path = matrix_path

		# 获取目录中的所有文件列表
		file_list = os.listdir(directory_path)
		matrix_list = []

		# 遍历文件列表，输出每个文件名
		for file_name in file_list:
			if file_name.endswith(".npy"):
				abs_file_name = os.path.join(matrix_path, file_name)
				matrix_list.append(np.load(abs_file_name))

		return matrix_list

	def save_dense_matrix(self, matrix_list, matrix_path):
		for index in range(len(matrix_list)):
			name = matrix_path + self.matrix_prefix + str(index) + ".npy"
			np.save(name, matrix_list[index])

		return

	def convert_sp2dense(self, input_list):
		dense_list = []
		for mat in input_list:
			dense_mat = mat.toarray().astype(np.int16)
			dense_list.append(dense_mat)
		return dense_list

	def compute_adj_with_matrix(self, x, edge_index, edge_weight, max_order=2, matrix_path=None):

		if self.order_matrix == None:
			if matrix_path != None:
				self.matrix_path = matrix_path
				print("loading dense matrix from disk...")
				sp_matrix_list = self.load_sparse_matrix(self.matrix_path)
				self.order_matrix = self.convert_sp2dense(sp_matrix_list)
			# self.adj_matrix = self.order_matrix[0]
			else:
				self.adj_matrix = None
				self.order_matrix = []
				self.adj_matrix = torch.zeros([len(x), len(x)]).int()
				for edge, weight in zip(edge_index.T, edge_weight):
					index_from = edge[0]
					index_to = edge[1]
					# input(edge[0] + "\t" + edge[1])
					self.adj_matrix[index_from, index_to] = 1

				# self.sp_matrix = self.adj_matrix
				self.sp_matrix = self.adj_matrix.float().to_sparse()

				for index in range(max_order):
					if index == 0:
						self.order_matrix.append(self.sp_matrix)
						continue
					else:
						tmp_matrix = self.order_matrix[-1]
						# out_matrix = torch.matmul(tmp_matrix, self.adj_matrix)
						out_matrix = torch.sparse.mm(tmp_matrix, self.sp_matrix)
						# out_matrix = out_matrix()

						self.order_matrix.append(out_matrix)

				dense_matrix = []
				for sp_m in self.order_matrix:
					d_m = sp_m.to_dense()
					np_d_m = d_m.numpy()
					del d_m
					dense_matrix.append(np_d_m)

				del self.order_matrix
				self.order_matrix = dense_matrix
		# self.save_sparse_matrix(self.order_matrix, self.matrix_path)
		# self.save_dense_matrix(self.order_matrix, self.matrix_path)

		else:
			pass

		self.adj_matrix = self.order_matrix[0]
		return self.order_matrix

	def compute_adj_entities(self, sentence):
		# init a probability transformation matrix
		for token, idx in sentence:
			token

		return

	def random_neighbor(self, node_id):
		neighbor_np = self.adj_matrix[node_id, ::]
		adj_neighbor = np.where(neighbor_np >= 1)
		idx = np.random.choice(adj_neighbor)
		return idx

	def is_neighbor(self, from_id, to_id):
		value = self.adj_matrix[from_id, to_id]
		result = value >= 1
		return result

	@njit(nogil=True)
	def random_walk(self, walk_length, p, q, t):
		"""sample a random walk starting from t
		"""
		# Normalize the weights to compute rejection probabilities
		max_prob = max(1 / p, 1, 1 / q)
		prob_0 = 1 / p / max_prob
		prob_1 = 1 / max_prob
		prob_2 = 1 / q / max_prob

		# Initialize the walk
		walk = np.empty(walk_length, dtype=np.int)
		walk[0] = t
		walk[1] = self.random_neighbor(t)

		for j in range(2, walk_length):
			while True:
				new_node = self.random_neighbor(walk[j - 1])
				r = np.random.rand()
				if new_node == walk[j - 2]:
					# back to the previous node
					if r < prob_0:
						break
				elif self.is_neighbor(walk[j - 2], new_node):
					# distance 1
					if r < prob_1:
						break
				elif r < prob_2:
					# distance 2
					break
			walk[j] = new_node

		return walk

	def top_n_indices(self, arr, n):
		result = arr.argsort()[-n:][::-1]
		return result

	# indices = np.argpartition(arr, -n)[-n:]
	# return indices[np.argsort(-arr[indices])]

	def exp_sum_weight(self, path_list, max_entities, init_weight=1, decay_rate=.9):
		result_matrix = np.zeros(len(path_list[0]))
		weight = init_weight
		for path_matrix in path_list:
			weight = weight * decay_rate
			result_matrix = result_matrix + path_matrix * weight

		selected_nodes = self.top_n_indices(result_matrix, max_entities)

		return selected_nodes

	def gumbel_exp_sum_weight(self, path_list, max_entities, init_weight=1, decay_rate=.9):
		result_matrix = np.zeros(len(path_list[0]))
		weight = init_weight
		for path_matrix in path_list:
			weight = weight * decay_rate
			result_matrix = result_matrix + path_matrix * weight

		# selected_nodes = self.top_n_indices(result_matrix, max_entities)
		selected_nodes = self.gumbel_softmax_sample(max_entities, result_matrix, decay_rate)

		return selected_nodes

	def gumbel_exp_sum_weight_dev(self, path_list, max_entities, original_nodes, init_weight=1, decay_rate=.1):

		if len(path_list) == 0:
			sampled_data = np.random.sample([max_entities]) * len(self.adj_matrix)
			sampled_data = sampled_data.astype(np.int)
			return sampled_data

		result_matrix = np.zeros(len(path_list[0]))
		weight = init_weight
		index = 1
		for path_matrix in path_list:
			weight = 1 / self.factorial_np(index)
			index = index + 1
			result_matrix = result_matrix + path_matrix * weight

		# selected_nodes = self.top_n_indices(result_matrix, max_entities)
		selected_nodes = self.gumbel_softmax_sample_dev(max_entities, self.normalize(result_matrix))

		return selected_nodes

	def construct_sub_matrix(self, selected_nodes):
		sub_matrix = self.adj_matrix[::, selected_nodes][selected_nodes, ::]
		return sub_matrix

	def average_longest_path_lengths(self, adj_matrix):
		# 使用邻接矩阵创建图
		graph = nx.from_numpy_array(adj_matrix)

		# 计算所有节点对之间的最短路径长度
		all_pairs_shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(graph))

		# 计算所有最短路径长度的总和
		total_path_lengths = 0
		max_path_length = 0
		num_paths = 0
		for source, paths in all_pairs_shortest_path_lengths.items():
			for target, length in paths.items():
				total_path_lengths += length
				if length > max_path_length:
					max_path_length = length
				num_paths += 1

		# 计算平均路径长度
		average_path_length = total_path_lengths / num_paths

		return average_path_length, max_path_length

	def factorial_np(self, n):
		return np.prod(np.arange(1, n + 1))

	def normalize(self, input_tensor):
		"""
		epsilon = 1e-10
		mean = input_tensor.mean()
		var = input_tensor.var()
		min = input_tensor.min()
		max = input_tensor.max()
		range = max - min
		obj = None

		std = obj.sqrt(var + epsilon)

		# 对输入张量进行普通归一化
		exp_result = obj.exp(input_tensor)
		output_tensor = exp_result / obj.sum(exp_result)
		# obj.sum(obj.exp(input_tensor))
		# output_tensor = (input_tensor - mean) / std
		# output_tensor = 1 / (1 + obj.exp(-output_tensor))
		"""
		if isinstance(input_tensor, np.ndarray):
			obj = np
		else:
			obj = torch

		sqrt_data = obj.sqrt(input_tensor)

		# 归一化
		min_sqrt = np.min(sqrt_data)
		max_sqrt = np.max(sqrt_data)
		if max_sqrt == 0 or max_sqrt - min_sqrt == 0:
			output_tensor =  input_tensor * 0 + 1/len(input_tensor)
			return output_tensor
		normalized_data = (sqrt_data - min_sqrt) / (max_sqrt - min_sqrt)
		output_tensor = normalized_data / normalized_data.sum()

		return output_tensor

	def max_path_sample(self, nodes, max_entities=10):
		result_nodes, result_edge = None, None
		path_list = []
		nodes = np.array(nodes)
		for order_matrix in self.order_matrix:
			sub_matrix = order_matrix[::, nodes]
			ipt_sub_matrix = order_matrix[nodes, ::]
			# opt_path_matrix = np.sum(sub_matrix, -1)
			opt_path_matrix = sub_matrix.sum(-1)
			# ipt_path_matrix = np.sum(ipt_sub_matrix, 0)
			ipt_path_matrix = ipt_sub_matrix.sum(0)
			path_matrix = self.normalize(opt_path_matrix * ipt_path_matrix)
			if not isinstance(path_matrix, np.ndarray):
				path_matrix = path_matrix.numpy()
			path_list.append(path_matrix)

		entities = self.exp_sum_weight(path_list, max_entities)

		return entities

	def max_path_sample_with_gumbel_connectivity(self, nodes, max_entities=10):
		result_nodes, result_edge = None, None
		path_list = []
		nodes = np.array(nodes)
		if len(nodes) != 0:
			for order_matrix in self.order_matrix:
				sub_matrix = order_matrix[nodes, ::]
				ipt_sub_matrix = order_matrix[::, nodes]
				# opt_path_matrix = np.sum(sub_matrix, -1)
				opt_path_matrix = sub_matrix.sum(0)
				# ipt_path_matrix = np.sum(ipt_sub_matrix, 0)
				ipt_path_matrix = ipt_sub_matrix.sum(-1)
				path_matrix = self.normalize(opt_path_matrix * ipt_path_matrix)
				# path_matrix = opt_path_matrix * ipt_path_matrix
				if not isinstance(path_matrix, np.ndarray):
					path_matrix = path_matrix.numpy()
				path_list.append(path_matrix)

		entities = self.gumbel_exp_sum_weight_dev(path_list, max_entities, nodes)

		return entities


	def max_path_sample_with_gumbel(self, nodes, max_entities=10):
		result_nodes, result_edge = None, None
		path_list = []
		nodes = np.array(nodes)

		for order_matrix in self.order_matrix:
			opt_sub_matrix = order_matrix[::, nodes]
			ipt_sub_matrix = order_matrix[nodes, ::]
			# opt_path_matrix = np.sum(sub_matrix, -1)
			opt_path_matrix = opt_sub_matrix.sum(-1)
			# ipt_path_matrix = np.sum(ipt_sub_matrix, 0)
			ipt_path_matrix = ipt_sub_matrix.sum(0)
			path_matrix = self.normalize(opt_path_matrix * ipt_path_matrix)
			if not isinstance(path_matrix, np.ndarray):
				path_matrix = path_matrix.numpy()
			path_list.append(path_matrix)

		entities = self.gumbel_exp_sum_weight(path_list, max_entities)

		return entities

	def get_values(self, csr_matrix_obj, row_indices, col_indices):
		# csr_matrix_obj = csr_matrix(matrix)

		# 获取指定行的视图
		selected_rows = csr_matrix_obj[row_indices, :]

		# 获取指定列的视图
		selected_cols = csr_matrix_obj[:, col_indices]

		# 返回指定行和列的值
		return selected_rows[:, col_indices].toarray()

	def max_path_sample_embedding_with_gumbel(self, nodes, max_entities=10):
		result_nodes, result_edge = None, None
		path_list = []
		nodes = np.array(nodes)

		for order_matrix in self.order_matrix:
			sub_matrix = order_matrix[::, nodes]
			ipt_sub_matrix = order_matrix[nodes, ::]
			# opt_path_matrix = np.sum(sub_matrix, -1)
			opt_path_matrix = sub_matrix.sum(-1)
			# ipt_path_matrix = np.sum(ipt_sub_matrix, 0)
			ipt_path_matrix = ipt_sub_matrix.sum(0)
			path_matrix = self.normalize(opt_path_matrix * ipt_path_matrix)
			if not isinstance(path_matrix, np.ndarray):
				path_matrix = path_matrix.numpy()
			path_list.append(path_matrix)

		entities = self.gumbel_exp_sum_weight(path_list, max_entities)

		return entities

	def get_words_from_list(self, input):
		result = []
		for item in input:
			result.append(self.get_words_from_id(item))
		return result

	def gumbel_softmax_sample(self, n, logits, temperature=0.01):
		noise_coefficient = .1
		mean = logits.mean()
		max = logits.max()
		min = logits.min()
		ae_gap = max - mean
		ea_gap = mean - min
		ai_gap = max - min
		float_gap = np.min([ea_gap, ai_gap])
		logistic = 1 / (1 + np.exp(-logits))
		gumbel_noise = -np.log(-np.log(np.random.rand(len(logistic))) + 1e-20)

		range_gumbel = gumbel_noise.max() - gumbel_noise.min()
		scale_coefficient = float_gap / range_gumbel
		gumbel_noise = gumbel_noise * scale_coefficient

		gumbel_logits = (logistic + gumbel_noise) / temperature
		probs = np.exp(gumbel_logits) / np.sum(np.exp(gumbel_logits))

		samples = np.random.choice(len(probs), n, replace=False, p=probs)
		str_entities = self.get_words_from_list(samples)
		return samples

	def gumbel_softmax_sample_dev(self, n, logits, temperature=0.01):
		noise_coefficient = .1
		one_sigma = .6526
		mean = logits.mean()
		max = logits.max()
		min = logits.min()
		ae_gap = max - mean
		ea_gap = mean - min
		ai_gap = max - min
		float_gap = np.min([ea_gap, ae_gap])
		logistic = self.normalize(logits) # logistic = 1 / (1 + np.exp(-logits))
		gumbel_noise = -np.log(-np.log(np.random.rand(len(logistic))) + 1e-20)

		# range_gumbel = gumbel_noise.max() - gumbel_noise.min()
		scale_coefficient = float_gap * logistic # scale_coefficient = float_gap / range_gumbel
		gumbel_noise = gumbel_noise * scale_coefficient
		# temperature = len(logits)
		# scaled_max = self.find_scaling_factor(n, one_sigma)
		#  scaled_max is 135.33163842241694 given N is 128 and sigma is .6526
		scaled_max = 135.33163842241694
		temperature = scaled_max / logistic.max()
		gumbel_logits = (logistic + gumbel_noise) * (temperature)
		probs = np.exp(gumbel_logits) / np.sum(np.exp(gumbel_logits))

		samples = np.random.choice(len(probs), n, replace=False, p=probs)
		# str_entities = self.get_words_from_list(samples)
		return samples

	def find_scaling_factor(self, N, sigma):
		# 定义 Softmax 函数
		x_solution = -N * np.log(1 - sigma)
		return x_solution # x is 135.33163842241694 given N is 128 and sigma is .6526

	def input_degree(self, entities):
		ipt_dgr = np.sum(self.adj_matrix, 0)
		ipt_dgr = ipt_dgr * entities
		return ipt_dgr

	def output_degree(self, entities):
		opt_dgr = np.sum(self.adj_matrix, 1)
		opt_dgr = opt_dgr * entities
		return opt_dgr

	def organize_vector(self, entities):
		length = self.adj_matrix.shape[0]

		init_vec = np.zeros([length])
		for ent in entities:
			init_vec[ent] = 1
		return init_vec

	def compute_paths(self, init_ent, target_ent, sample_number):
		path_list = []
		nodes = np.array(init_ent)
		if len(nodes) == 0:
			num = self.adj_matrix.shape[0]
			nodes = np.linspace(0, num - 1, num).astype(np.int)
		for order_matrix in self.order_matrix:
			init_matrix = order_matrix[nodes, ::]
			path_number_matrix = init_matrix[::, target_ent]
			norm_path_number_matrix = self.normalize(path_number_matrix)
			ipt_path_matrix = np.sum(order_matrix[::, target_ent], 0)
			opt_path_matrix = np.sum(order_matrix[target_ent, ::], 1)
			norm_deg_path_matrix = self.normalize(ipt_path_matrix * opt_path_matrix)
			# we should add norm operation to ipt_path and opt_path
			path_list.append(norm_path_number_matrix * norm_deg_path_matrix)
			"""
			opt_path_matrix = np.sum(init_matrix, -1)
			ipt_path_matrix = np.sum(init_matrix, 0)
			"""
		sum_matrix = np.zeros_like(path_list[0])
		rate = 0.9
		for path_matrix in path_list:
			sum_matrix = sum_matrix + path_matrix * rate
			rate = rate * rate

		result = self.top_n_indices(sum_matrix.sum(-1), sample_number)

		return result

	def naive_mps_sample(self, init_adj_edge, source_entities, mid_entities, sample_number):
		result = []
		length = self.adj_matrix.shape[0]
		if len(source_entities) == 0:
			return result
		per_sample_number = sample_number // len(source_entities)
		for index in range(len(init_adj_edge)):
			entity = source_entities[index]
			str_head_ent = self.get_words_from_id(entity)
			edges = init_adj_edge[index]
			# init_vec = np.zeros([length])
			init_list = []

			if len(edges) == 0:
				rels = sorted(list(self.mp_all.get(str_head_ent, [])), key=lambda x: x[2],
							  reverse=True)[:per_sample_number]

				if rels is not None:
					for rel in rels:
						str_tail_ent = rel[0]
						result.append([str_head_ent, rel, str_tail_ent])
				else:
					result.append([str_head_ent, rels, None])
				continue

			for edge in edges:
				str_tail_ent = edge[0]
				id_tail_ent = self.get_id_from_words(str_tail_ent)
				# init_vec[id_tail_ent] = init_vec[id_tail_ent] + 1
				init_list.append(id_tail_ent)

			sample_ent_index = self.compute_paths(init_list, mid_entities, per_sample_number)
			for ent_index in sample_ent_index:
				ent_id = init_list[ent_index]
				ent_str = self.get_words_from_id(ent_id)
				rel = edges[ent_index]
				str_tail_ent = rel[0]
				result.append([str_head_ent, rel, str_tail_ent])
		return result

	def sample_relation_from_entity(self, source_entities, mid_entities, sample_number=-1):
		if sample_number == -1:
			sample_number = len(source_entities) * 2
		input_rel_set = []
		result = []
		tmp_source_entities = source_entities.copy()
		for ent in source_entities:
			rels = self.mp_all.get(self.get_words_from_id(ent))

			list_rels = [] if rels is None else [_ for _ in rels]
			input_rel_set.append(list_rels)

		result = self.naive_mps_sample(input_rel_set, tmp_source_entities, mid_entities, sample_number)

		return result

	def upscale_sample(self, init_set, mid_set, transform_matrix, sample_number):
		ent_number = len(init_set)
		per_sample_number = sample_number // ent_number
		sample = []
		for index in range(ent_number):
			rel_index_set = self.top_n_indices(transform_matrix[index, ::], per_sample_number)
			from_ent = init_set[index]
			str_source_ent = self.get_words_from_id(from_ent)
			rel_set = self.mp_all.get(str_source_ent)
			rel_set = rel_set if rel_set is not None else None
			for rel in rel_set:  # source 2 middle
				str_tail_ent = rel[0]
				for rel_index_value in rel_index_set:
					if self.get_id_from_words(str_tail_ent) == rel_index_value:
						sample.append([str_source_ent, rel, str_tail_ent])

		return sample

	def downscale_sample(self, init_set, mid_set, transform_matrix, sample_number):

		ent_number = len(init_set)
		per_sample_number = sample_number // ent_number
		sample = []
		for index in range(ent_number):
			rel_index_set = self.top_n_indices(transform_matrix[index, ::], per_sample_number)
			from_ent = mid_set[index]
			str_source_ent = self.get_words_from_id(from_ent)
			rel_set = self.mp_all.get(str_source_ent)
			rel_set = rel_set if rel_set is not None else []
			for rel in rel_set:
				str_tail_ent = rel[0]  # m2s 中 尾节点 str
				for rel_index_value in rel_index_set:
					str_top_n = self.get_words_from_id(rel_index_value)
					if str_tail_ent == str_top_n:
						sample.append([str_source_ent, rel, str_tail_ent])

		return sample

	def compute_depart_return_path(self, source, mid_entities, source_mp=None, mid_mp=None):
		# source_local_id = OrderedDict()
		# source_local_str = OrderedDict()
		source_local_dict = OrderedDict()
		mid_local_dict = OrderedDict()
		for index in range(self.local_graph_words_encode_idx):
			if index < self.source_entities_limit:
				source_local_dict[self.local_graph_words_to_id.inverse[index]] = index
				continue
			entity_str = self.local_graph_words_to_id.inverse[index]
			# entitiy_global_id = self.get_id_from_words(entity_str)
			if entity_str in mid_entities:
				mid_local_dict[self.local_graph_words_to_id.inverse[index]] = index

		for index in range(self.source_entities_limit, self.local_graph_words_encode_idx):
			entity_str = self.local_graph_words_to_id.inverse[index]
				# entitiy_global_id = self.get_id_from_words(entity_str)
			if entity_str not in mid_entities:
				mid_local_dict[self.local_graph_words_to_id.inverse[index]] = index
			if len(mid_local_dict.keys()) == len(mid_entities):
				break

		# translate the local id to mp_all id
		source_mp_all_id = self.translate_words2id(source_local_dict.keys())
		mid_mp_all_id = self.translate_words2id(mid_local_dict.keys())

		depart_matrix = []
		return_matrix = []

		# all_matrix = self.adj_matrix[source_mp_all_id][:, mid_mp_all_id]

		depart_path_list = []
		return_path_list = []

		order = len(self.order_matrix)

		for order_matrix in self.order_matrix:
			depart_matrix = order_matrix[source_mp_all_id][:, mid_mp_all_id]
			return_matrix = order_matrix[mid_mp_all_id][:, source_mp_all_id]


			# norm_depart_path_number_matrix = self.normalize(depart_matrix)
			# norm_return_path_number_matrix = self.normalize(return_matrix)
			depart_path_list.append(depart_matrix)
			return_path_list.append(return_matrix)

			"""
			opt_path_matrix = np.sum(init_matrix, -1)
			ipt_path_matrix = np.sum(init_matrix, 0)
			"""

		all_depart_matrix = np.sum(depart_path_list, axis=0)
		all_return_matrix = np.sum(return_path_list, axis=0)

		return all_depart_matrix, all_return_matrix

	def save_sample_pre_process_result(self, input_data):

		return

	def load_sample_pre_process_result(self, input_data):

		return

	def mps(self, init, end, transform_matrix):
		length = len(init)
		# transform_matrix
		return

	def translate_id2words(self, corpus):
		words = []
		for word in corpus:
			words.append(self.get_words_from_id(word))
		return words

	def translate_words2id(self, words):
		ids = []
		for word in words:
			ids.append(self.get_id_from_words(word))
		return ids

	def compute_edge_to_ent(self, edge, ent):

		return

	def get_graph_id_from_words(self, words, is_add=False):
		if self.graph_words_to_id.get(words) is None and is_add:
			with self.semaphore:
				self.graph_words_to_id[words] = self.graph_words_encode_idx
				self.graph_words_encode_idx += 1
				self.graph_real_words_seq.append(self.get_id_from_words(words))
		return self.graph_words_to_id.get(words)


	def get_id_from_words(self, words, is_add=False):
		with self.semaphore:
			if self.words_to_id.get(words) is None and is_add:
				self.words_to_id[words] = self.words_encode_idx
				self.words_encode_idx += 1
			return_result = self.words_to_id.get(words)
		return return_result

	def get_local_id_from_words(self, words, is_add=False):
		with self.semaphore:
			if self.local_graph_words_to_id.get(words) is None and is_add:
				self.local_graph_words_to_id[words] = self.local_graph_words_encode_idx
				self.local_graph_words_encode_idx += 1
			return_result = self.local_graph_words_to_id.get(words)
		return return_result

	def encode_graph_sorted_index(self, mp):

		x_index_id = []
		edge_index = []
		edge_weight = []
		self.graph_words_encode_idx = 0
		self.graph_words_to_id.clear()
		self.graph_real_words_seq = []
		real_word_id = []
		sorted_item = sorted(mp.items(), key=lambda x: x[0])

		for item in sorted_item:
			key = item[0]
			values = item[1]
			st_id = self.get_graph_id_from_words(key, is_add=True)
			# real_st_id = self.get_id_from_words(key, is_add=True)
			# real_word_id.append(real_st_id)
			x_index_id.append(st_id)  # 代表所有出现在 sentence 中的 word id
			for value in values:
				to_words = value[0]
				to_relation = value[1]
				to_weight = value[2]
				# real_id = self.get_id_from_words(to_words, is_add=False)
				# real_word_id.append(real_id)
				alias_id = self.get_graph_id_from_words(to_words, is_add=True)
				ed_id = alias_id
				# ed_id = self.get_id_from_words(to_words, is_add=True)

				# 暂定建立双向边
				edge_index.append([st_id, ed_id])
				edge_weight.append(to_weight)
				edge_index.append([ed_id, st_id])
				edge_weight.append(to_weight)
		# 建立每个 id 对应词与词向量映射关系
		# real_word_id = list(set(real_word_id))
		x = [self.get_features_from_words(self.get_words_from_id(self.graph_real_words_seq[i])) for i in
			 range(self.local_graph_words_encode_idx)]
		x_index = torch.zeros(len(x), dtype=torch.bool)
		x_index[x_index_id] = 1

		return torch.tensor(x), x_index, torch.tensor(edge_index, dtype=torch.long).t(), torch.tensor(edge_weight)

	def encode_sorted_index(self, mp):

		x_index_id = []
		edge_index = []
		edge_weight = []
		self.words_encode_idx = 0
		self.words_to_id.clear()

		sorted_item = sorted(mp.items(), key=lambda x: x[0])

		# 建边
		for item in sorted_item:
			key = item[0]
			values = item[1]
			st_id = self.get_id_from_words(key, is_add=True)
			x_index_id.append(st_id)  # 代表所有出现在 sentence 中的 word id
			for value in values:
				to_words = value[0]
				to_relation = value[1]
				to_weight = value[2]

				ed_id = self.get_id_from_words(to_words, is_add=True)

				# 暂定建立双向边
				edge_index.append([st_id, ed_id])
				edge_weight.append(to_weight)
				edge_index.append([ed_id, st_id])
				edge_weight.append(to_weight)
		# 建立每个 id 对应词与词向量映射关系
		x = [self.get_features_from_words(self.get_words_from_id(i)) for i in range(self.words_encode_idx)]
		x_index = torch.zeros(len(x), dtype=torch.bool)
		x_index[x_index_id] = 1

		return torch.tensor(x), x_index, torch.tensor(edge_index, dtype=torch.long).t(), torch.tensor(edge_weight)

	def encode_index(self, mp):
		"""
		建立一个 words to id 的映射表，使用 bidict 可以双向解析
		"""
		x_index_id = []
		edge_index = []
		edge_weight = []
		self.words_encode_idx = 0
		self.graph_words_to_id.clear()

		# 建边
		for key, values in mp.items():
			st_id = self.get_id_from_words(key, is_add=True)
			x_index_id.append(st_id)  # 代表所有出现在 sentence 中的 word id
			for value in values:
				to_words = value[0]
				to_relation = value[1]
				to_weight = value[2]

				ed_id = self.get_id_from_words(to_words, is_add=True)

				# 暂定建立双向边
				edge_index.append([st_id, ed_id])
				edge_weight.append(to_weight)
				edge_index.append([ed_id, st_id])
				edge_weight.append(to_weight)
		# 建立每个 id 对应词与词向量映射关系
		x = [self.get_features_from_words(self.get_words_from_id(i)) for i in range(self.words_encode_idx)]
		x_index = torch.zeros(len(x), dtype=torch.bool)
		x_index[x_index_id] = 1

		return torch.tensor(x), x_index, torch.tensor(edge_index, dtype=torch.long).t(), torch.tensor(edge_weight)

	def encode_local_index(self, mp):
		source_mp, mid_mp = mp
		x_index_id = []
		edge_index = []
		edge_weight = []
		self.local_graph_words_encode_idx = 0
		self.source_entities_limit = 0
		self.local_graph_words_to_id.clear()

		# sorted_item = sorted(mp.items(), key=lambda x: x[0])

		# 建边
		for key, values in source_mp.items():
			st_id = self.get_local_id_from_words(key, is_add=True)
			x_index_id.append(st_id)  # 代表所有出现在 sentence 中的 word id
			for value in values:
				to_words = value[0]
				to_relation = value[1]
				to_weight = value[2]

				ed_id = self.get_local_id_from_words(to_words, is_add=True)

				# 暂定建立双向边
				edge_index.append([st_id, ed_id])
				edge_weight.append(to_weight)
				edge_index.append([ed_id, st_id])
				edge_weight.append(to_weight)

		self.source_entities_limit = self.local_graph_words_encode_idx
		# this will be an issue when intersection words exist in mid and source mp
		for key, values in mid_mp.items():
			st_id = self.get_local_id_from_words(key, is_add=True)
			x_index_id.append(st_id)  # 代表所有出现在 sentence 中的 word id
			for value in values:
				to_words = value[0]
				to_relation = value[1]
				to_weight = value[2]

				ed_id = self.get_local_id_from_words(to_words, is_add=True)

				# 暂定建立双向边
				edge_index.append([st_id, ed_id])
				edge_weight.append(to_weight)
				edge_index.append([ed_id, st_id])
				edge_weight.append(to_weight)
		# 建立每个 id 对应词与词向量映射关系
		x = [self.get_features_from_words(self.get_local_words_from_id(i)) for i in range(self.local_graph_words_encode_idx)]
		x_index = torch.zeros(len(x), dtype=torch.bool)
		x_index[x_index_id] = 1

		return torch.tensor(x), x_index, torch.tensor(edge_index, dtype=torch.long).t(), torch.tensor(edge_weight)

	"""
	from torch_geometric.utils import to_dense_adj
	to_dense_adj

	def dfs(adj, visited, v, length):
		visited[v] = True
		max_length = length
		for i in range(len(adj)):
			if adj[v][i] == 1 and not visited[i]:
				max_length = max(max_length, dfs(adj, visited, i, length + 1))
		visited[v] = False
		return max_length
	
	def longest_path(adj):
		n = len(adj)
		visited = [False] * n
		max_path_length = 0
		for i in range(n):
			max_path_length = max(max_path_length, dfs(adj, visited, i, 1))
		return max_path_length
	
	"""

	def merge_mp(self, source, mid_mp):
		merged_defaultdict = defaultdict(list)

		# 合并第一个defaultdict
		for key, values in source.items():
			merged_defaultdict[key].extend(values)

		# 合并第二个defaultdict
		for key, values in mid_mp.items():
			merged_defaultdict[key].extend(values)

		return merged_defaultdict

	def get_submp_by_sentences(self, sentences: list, is_merge=False):
		"""
		获取 conceptnet 中的一个子图
		:param sentences: 一个列表，如 ["I am a student.", "Hi!"]
		:param is_merge: 是否合并 sentences 中每个元素的结果
		:return: 子图 mp
		"""

		def get_submp_by_one(mp_all, sent, n_gram=1, stop_words=[]):
			lemmatizer = WordNetLemmatizer()
			mp_sub = defaultdict(set)
			sent = sent.strip(',|.|?|;|:|!').lower()
			tokens = self.tokenizer.tokenize(sent)
			# 这里加 '#' 是为了防止 tokenizer 与传统根据分割方法 n_gram 在一起
			tokens += ['#'] + re.sub('[^a-zA-Z0-9,]', ' ', sent).split(' ')
			for gram in range(1, n_gram + 1):
				start, end = 0, gram
				while end <= len(tokens):
					# n_gram 将要连接的单词，中间需要用 _ 分割
					q_words = '_'.join(tokens[start:end])
					start, end = start + 1, end + 1

					if gram == 1 and q_words in stop_words:
						# 如果 q_words 是停用词
						continue
					if q_words.find('#') != -1:
						# 如果其中有 # 号
						continue
					if gram == 1:
						q_words = lemmatizer.lemmatize(q_words, pos='n')  # 还原词性为名词

					if mp_all.get(q_words) is not None and mp_sub.get(q_words) is None:
						# 如果 mp_all 中存在 q_words 并且 mp_sub 中不存在 q_words
						mp_sub[q_words] |= mp_all[q_words]
			return mp_sub

		if is_merge:
			sent = ' '.join(sentences)
			sentences = [sent]

		res = []
		for i in sentences:
			res.append(get_submp_by_one(self.mp_all, i, self.args['n_gram'],
										stop_words=self.args['reduce_noise_args']['stop_words']))

		return res


	def get_local_submp_by_sentences(self, sentences: list, is_merge=False):
		"""
		获取 conceptnet 中的一个子图
		:param sentences: 一个列表，如 ["I am a student.", "Hi!"]
		:param is_merge: 是否合并 sentences 中每个元素的结果
		:return: 子图 mp
		"""

		def get_submp_by_one(mp_all, sent, n_gram=1, stop_words=[]):
			lemmatizer = WordNetLemmatizer()
			mp_sub = defaultdict(set)
			sent = sent.strip(',|.|?|;|:|!').lower()
			tokens = self.tokenizer.tokenize(sent)
			# 这里加 '#' 是为了防止 tokenizer 与传统根据分割方法 n_gram 在一起
			tokens += ['#'] + re.sub('[^a-zA-Z0-9,]', ' ', sent).split(' ')
			for gram in range(1, n_gram + 1):
				start, end = 0, gram
				while end <= len(tokens):
					# n_gram 将要连接的单词，中间需要用 _ 分割
					q_words = '_'.join(tokens[start:end])
					start, end = start + 1, end + 1

					if gram == 1 and q_words in stop_words:
						# 如果 q_words 是停用词
						continue
					if q_words.find('#') != -1:
						# 如果其中有 # 号
						continue
					if gram == 1:
						q_words = lemmatizer.lemmatize(q_words, pos='n')  # 还原词性为名词

					if mp_all.get(q_words) is not None and mp_sub.get(q_words) is None:
						# 如果 mp_all 中存在 q_words 并且 mp_sub 中不存在 q_words
						mp_sub[q_words] |= mp_all[q_words]
			return mp_sub

		def get_mid_submp_by_one(mp_all, sent, n_gram=1, stop_words=[]):
			# lemmatizer = WordNetLemmatizer()
			mp_sub = defaultdict(set)
			sent = sent.strip(',|.|?|;|:|!').lower()
			tokens = sent.split(' ')
			# 这里加 '#' 是为了防止 tokenizer 与传统根据分割方法 n_gram 在一起
			# tokens += ['#'] + re.sub('[^a-zA-Z0-9,]', ' ', sent).split(' ')
			for token in tokens:
				if mp_all.get(token) is not None:
						# 如果 mp_all 中存在 q_words 并且 mp_sub 中不存在 q_words
						mp_sub[token] |= mp_all[token]
			if len(mp_sub) == len(tokens):
				return mp_sub

			lemmatizer = WordNetLemmatizer()
			sent = sent.strip(',|.|?|;|:|!').lower()
			tokens = self.tokenizer.tokenize(sent)
			# 这里加 '#' 是为了防止 tokenizer 与传统根据分割方法 n_gram 在一起
			tokens += ['#'] + re.sub('[^a-zA-Z0-9,]', ' ', sent).split(' ')
			for gram in range(1, n_gram + 1):
				start, end = 0, gram
				while end <= len(tokens):
					# n_gram 将要连接的单词，中间需要用 _ 分割
					q_words = '_'.join(tokens[start:end])
					start, end = start + 1, end + 1

					if gram == 1 and q_words in stop_words:
						# 如果 q_words 是停用词
						continue
					if q_words.find('#') != -1:
						# 如果其中有 # 号
						continue
					if gram == 1:
						q_words = lemmatizer.lemmatize(q_words, pos='n')  # 还原词性为名词

					if mp_all.get(q_words) is not None and mp_sub.get(q_words) is None:
						# 如果 mp_all 中存在 q_words 并且 mp_sub 中不存在 q_words
						mp_sub[q_words] |= mp_all[q_words]
					if len(mp_sub) == len(tokens):
						return mp_sub
			return mp_sub


		if is_merge:
			sent = ' '.join(sentences)
			sentences = [sent]

		res = []

		res.append(get_submp_by_one(self.mp_all, sentences[0], self.args['n_gram'],
										stop_words=self.args['reduce_noise_args']['stop_words']))
		res.append(get_mid_submp_by_one(self.mp_all, sentences[1], self.args['n_gram'],
										stop_words=self.args['reduce_noise_args']['stop_words']))

		return res



	def get_complex_submp_by_sentences(self, sentences, pub_ent, is_merge=False):
		"""
		获取 conceptnet 中的一个子图
		:param sentences: 一个列表，如 ["I am a student.", "Hi!"]
		:param is_merge: 是否合并 sentences 中每个元素的结果
		:return: 子图 mp
		"""

		def get_submp_by_one(mp_all, sent, n_gram=1, stop_words=[]):
			lemmatizer = WordNetLemmatizer()
			mp_sub = defaultdict(set)
			sent = sent.strip(',|.|?|;|:|!').lower()
			tokens = self.tokenizer.tokenize(sent)
			# 这里加 '#' 是为了防止 tokenizer 与传统根据分割方法 n_gram 在一起
			tokens += ['#'] + re.sub('[^a-zA-Z0-9,]', ' ', sent).split(' ')
			for gram in range(1, n_gram + 1):
				start, end = 0, gram
				while end <= len(tokens):
					# n_gram 将要连接的单词，中间需要用 _ 分割
					q_words = '_'.join(tokens[start:end])
					start, end = start + 1, end + 1

					if gram == 1 and q_words in stop_words:
						# 如果 q_words 是停用词
						continue
					if q_words.find('#') != -1:
						# 如果其中有 # 号
						continue
					if gram == 1:
						q_words = lemmatizer.lemmatize(q_words, pos='n')  # 还原词性为名词

					if mp_all.get(q_words) is not None and mp_sub.get(q_words) is None:
						# 如果 mp_all 中存在 q_words 并且 mp_sub 中不存在 q_words
						mp_sub[q_words] |= mp_all[q_words]
			return mp_sub

		if is_merge:
			sent = ' '.join(sentences)
			sentences = [sent]

		res = []
		for i in sentences:
			rels = get_submp_by_one(self.mp_all, i, self.args['n_gram'],
									stop_words=self.args['reduce_noise_args']['stop_words'])
			res.append(rels)

		mp_sub = res[0]
		for ent_id in pub_ent[0]:
			q_words = self.get_words_from_id(ent_id)
			# mp_sub = defaultdict(set)

			if self.mp_all.get(q_words) is not None:
				# 如果 mp_all 中存在 q_words 并且 mp_sub 中不存在 q_words
				mp_sub[q_words] |= self.mp_all[q_words]
		# res.append(mp_sub)
		return mp_sub

	def save_sparse_matrix(self, order_matrix, matrix_path):
		for index in range(len(order_matrix)):
			mat = order_matrix[index]
			sp_matrix = coo_matrix(mat)
			name = matrix_path + self.matrix_prefix + str(index) + ".npy"
			save_npz(name, sp_matrix)

		return

	def load_sparse_matrix(self, matrix_path):
		import os

		# 设置需要列举文件的目录路径
		directory_path = matrix_path

		# 获取目录中的所有文件列表
		file_list = os.listdir(directory_path)
		matrix_list = []

		# 遍历文件列表，输出每个文件名
		for file_name in file_list:
			if file_name.endswith(".npz"):
				abs_file_name = os.path.join(matrix_path, file_name)
				loaded_sparse_matrix = load_npz(abs_file_name)
				matrix_list.append(loaded_sparse_matrix)
		return matrix_list


class GraphUtils:
		def __init__(self):
			self.mp_all = defaultdict(set)  # 记录 id -> id 边的关系
			self.words_to_id = bidict()  # 将一个词映射为 id，仅在 encode_mp 时候构建
			self.words_encode_idx = 0  # 在 encode_mp 时构建
			self.conceptnet_numberbatch_en = dict()
			# 这里之所以不用 GPT2/Roberta tokenizer 是因为空格会被分割为 Ġ
			self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
			self.args = {
				'n_gram': 3,
				'mp_pickle_path': './conceptnet5/res_all.pickle',
				'conceptnet_numberbatch_en_path': './conceptnet5/numberbatch-en.txt',
				'reduce_noise_args': {
					# 白名单相对优先
					'relation_white_list': [],
					# ['/r/RelatedTo', '/r/IsA', '/r/PartOf', '/r/HasA', '/r/UsedFor', '/r/CapableOf', '/r/AtLocation', '/r/Causes', '/r/HasProperty'],
					'relation_black_list': ['/r/ExternalURL', '/r/Synonym', '/r/Antonym',
											'/r/DistinctFrom', '/r/dbpedia/genre', '/r/dbpedia/influencedBy'],
					'stop_words': ['a', 'able', 'about', 'across', 'after', 'all', 'almost', 'also', 'am', 'among',
								   'an',
								   'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'but', 'by', 'can',
								   'cannot',
								   'could', 'dear', 'did', 'do', 'does', 'either', 'else', 'ever', 'every', 'for',
								   'from',
								   'get', 'got', 'had', 'has', 'have', 'he', 'her', 'hers', 'him', 'his', 'how',
								   'however',
								   'i', 'if', 'in', 'into', 'is', 'it', 'its', 'just', 'least', 'let', 'like', 'likely',
								   'may', 'me', 'might', 'most', 'must', 'my', 'neither', 'no', 'nor', 'not', 'of',
								   'off',
								   'often', 'on', 'only', 'or', 'other', 'our', 'own', 'rather', 'said', 'say', 'says',
								   'she', 'should', 'since', 'so', 'some', 'than', 'that', 'the', 'their', 'them',
								   'then',
								   'there', 'these', 'they', 'this', 'tis', 'to', 'too', 'twas', 'us', 'wants', 'was',
								   'we',
								   'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will',
								   'with',
								   'would', 'yet', 'you', 'your'],
					'weight_limit': 1.5,
					'edge_count_limit': 100,  # 保留权重最大的 edge_count_limit 条边
				}
			}

		def load_mp_all_by_pickle(self, fpath):
			"""
            从 pickle 中加载 conceptnet 图
            :param fpath:
            :return:
            """
			graph_zip = None
			with open(fpath, 'rb') as f:
				graph_zip = pickle.load(f)
			self.mp_all, = graph_zip
			return graph_zip

		def reduce_graph_noise(self, is_overwrite=True):
			"""
            基于 relation type 以及 edge weight 降低图的噪声
            :param is_overwrite: 是否写入到 self.mp 中
            :return:
            """
			relation_white_list = self.args['reduce_noise_args']['relation_white_list']
			relation_black_list = self.args['reduce_noise_args']['relation_black_list']
			stop_words = self.args['reduce_noise_args']['stop_words']
			weight_limit = self.args['reduce_noise_args']['weight_limit']
			edge_count_limit = self.args['reduce_noise_args']['edge_count_limit']
			is_black_list = True  # 默认是开启黑名单的

			if len(relation_white_list) != 0:
				# 如果白名单里有则启用白名单
				is_black_list = False

			new_mp = defaultdict(set)  # 记录 id -> id 边的关系
			for key, values in self.mp_all.items():
				st_words = key
				if st_words in stop_words:
					# 停用词跳过
					continue

				# 取 values 按 edge_weight 从大到小排序以后的前 edge_count_limit 个（也可按概率选取）
				to_values = sorted(list(values), key=lambda x: x[2], reverse=True)
				edge_count = 0
				for value in to_values:
					to_words = value[0]
					to_relation = value[1]
					to_weight = value[2]
					if to_words in stop_words:
						# 停用词跳过
						continue
					if to_weight < weight_limit:
						# 边权较低的跳过
						continue
					if is_black_list:
						# 如果黑名单开启并且当前 relation 在黑名单里跳过
						if to_relation in relation_black_list:
							continue
					else:
						# 白名单下如果 relation 不在白名单里跳过
						if to_relation not in relation_white_list:
							continue
					new_mp[st_words].add((to_words, to_relation, to_weight))
					edge_count += 1
					if edge_count >= edge_count_limit:
						break

			if is_overwrite:
				self.mp_all = new_mp
			return new_mp

		def merge_graph_by_downgrade(self, is_overwrite=True):
			"""
            降级合并 mp 图，将形如 /c/en/apple/n 降级为 /c/en/apple，并省略 /c/en/
            :param is_overwrite: 是否将最终的结果直接写入 self 中的对象
            :return: 降级以后的 mp
            """
			new_mp = defaultdict(set)  # 记录 id -> id 边的关系
			refine_sent = lambda s: re.match('/c/en/([^/]+)', s).group(1)
			for key, values in self.mp_all.items():
				st_words = refine_sent(key)
				for value in values:
					to_words = refine_sent(value[0])
					to_relation = value[1]
					to_weight = value[2]
					new_mp[st_words].add((to_words, to_relation, to_weight))
			if is_overwrite:
				self.mp_all = new_mp
			return new_mp

		def init(self, is_load_necessary_data=True):
			"""
            load 部分数据初始化
            :return:
            """
			self.__init__()
			if is_load_necessary_data:
				self.load_mp_all_by_pickle(self.args['mp_pickle_path'])
				self.load_conceptnet_numberbatch(self.args['conceptnet_numberbatch_en_path'])

		def get_features_from_words(self, words):
			"""
            获取 words 的词向量
            :param words:
            :return:
            """
			words = standardized_uri('en', words).replace('/c/en/', '')
			res = self.conceptnet_numberbatch_en.get(words)
			if res is None:
				"""todo: 解决 OOV 问题，待定，暂时用 ## 代替"""
				# res = self.conceptnet_numberbatch_en.get('##')
				res = self.get_default_oov_feature()
			return res

		def get_default_oov_feature(self):
			# 默认的 oov feature
			return [0.0 for _ in range(300)]

		def load_conceptnet_numberbatch(self, fpath):
			"""
            从 numberbatch 中加载每个词的词向量
            :param fpath:
            :return:
            """
			if len(self.conceptnet_numberbatch_en) != 0:
				# 如果已经加载则不用管了
				return
			self.conceptnet_numberbatch_en.clear()
			with open(fpath, 'r', encoding='UTF-8') as f:
				n, vec_size = list(map(int, f.readline().split(' ')))
				print('load conceptnet numberbatch: ', n, vec_size)
				for i in range(n):
					tmp_data = f.readline().split(' ')
					words = str(tmp_data[0])
					vector = list(map(float, tmp_data[1:]))
					self.conceptnet_numberbatch_en[words] = vector
				print('load conceptnet numberbatch done!')

		def get_words_from_id(self, id):
			"""
            从 id 获取对应的 words
            :param id:
            :return:
            """
			return self.words_to_id.inverse.get(id)

		def get_id_from_words(self, words, is_add=False):
			"""
            从 words 获取其映射的 id
            :param words:
            :param is_add: 如果不存在是否加入进去
            :return:
            """
			if self.words_to_id.get(words) is None and is_add:
				self.words_to_id[words] = self.words_encode_idx
				self.words_encode_idx += 1
			return self.words_to_id.get(words)

		def encode_index(self, mp):
			"""
            建立一个 words to id 的映射表，使用 bidict 可以双向解析
            """
			x_index_id = []
			edge_index = []
			edge_weight = []
			self.words_encode_idx = 0
			self.words_to_id.clear()
			# 建边
			for key, values in mp.items():
				st_id = self.get_id_from_words(key, is_add=True)
				x_index_id.append(st_id)  # 代表所有出现在 sentence 中的 word id
				for value in values:
					to_words = value[0]
					to_relation = value[1]
					to_weight = value[2]

					ed_id = self.get_id_from_words(to_words, is_add=True)

					# 暂定建立双向边
					edge_index.append([st_id, ed_id])
					edge_weight.append(to_weight)
					edge_index.append([ed_id, st_id])
					edge_weight.append(to_weight)
			# 建立每个 id 对应词与词向量映射关系
			x = [self.get_features_from_words(self.get_words_from_id(i)) for i in range(self.words_encode_idx)]
			x_index = torch.zeros(len(x), dtype=torch.bool)
			x_index[x_index_id] = 1
			return torch.tensor(x), x_index, torch.tensor(edge_index, dtype=torch.long).t(), torch.tensor(edge_weight)

		def get_submp_by_sentences(self, sentences: list, is_merge=False):
			"""
            获取 conceptnet 中的一个子图
            :param sentences: 一个列表，如 ["I am a student.", "Hi!"]
            :param is_merge: 是否合并 sentences 中每个元素的结果
            :return: 子图 mp
            """

			def get_submp_by_one(mp_all, sent, n_gram=1, stop_words=[]):
				lemmatizer = WordNetLemmatizer()
				mp_sub = defaultdict(set)
				sent = sent.strip(',|.|?|;|:|!').lower()
				tokens = self.tokenizer.tokenize(sent)
				# 这里加 '#' 是为了防止 tokenizer 与传统根据分割方法 n_gram 在一起
				tokens += ['#'] + re.sub('[^a-zA-Z0-9,]', ' ', sent).split(' ')
				for gram in range(1, n_gram + 1):
					start, end = 0, gram
					while end <= len(tokens):
						# n_gram 将要连接的单词，中间需要用 _ 分割
						q_words = '_'.join(tokens[start:end])
						start, end = start + 1, end + 1

						if gram == 1 and q_words in stop_words:
							# 如果 q_words 是停用词
							continue
						if q_words.find('#') != -1:
							# 如果其中有 # 号
							continue
						if gram == 1:
							q_words = lemmatizer.lemmatize(q_words, pos='n')  # 还原词性为名词

						if mp_all.get(q_words) is not None and mp_sub.get(q_words) is None:
							# 如果 mp_all 中存在 q_words 并且 mp_sub 中不存在 q_words
							mp_sub[q_words] |= mp_all[q_words]
				return mp_sub

			if is_merge:
				sent = ' '.join(sentences)
				sentences = [sent]

			res = []
			for i in sentences:
				res.append(get_submp_by_one(self.mp_all, i, self.args['n_gram'],
											stop_words=self.args['reduce_noise_args']['stop_words']))
			return res


# %%
if __name__ == '__main__':
	import os
	
	os.chdir('..')
	graph = GraphUtils()
	graph.init(is_load_necessary_data=True)
	graph.merge_graph_by_downgrade()
	mp = graph.reduce_graph_noise(is_overwrite=False)
	# data = np.array(get_datas(
	#     get_data_from_task_2(
	#         '../SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Training Data/subtaskB_data_all.csv',
	#         '../SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Training Data/subtaskB_answers_all.csv'),
	#     get_data_from_task_2(
	#         '../SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Trial Data/taskB_trial_data.csv',
	#         '../SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Trial Data/taskB_trial_answer.csv')
	# ))
	# mp = graph.get_submp_by_sentences(['I am a students.', 'You have a apples and bananas'], is_merge=True)[0]
	# x, edge_index, edge_weight = graph.encode_index(mp)
