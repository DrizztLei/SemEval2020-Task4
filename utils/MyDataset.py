import os

import torch.utils.data
from torch_geometric.data import Data


# %%
class MyDataset(torch.utils.data.Dataset):
	def __init__(self, x, y):
		self.x = x
		self.y = y
	
	def __getitem__(self, item):
		return self.x[item], self.y[item]
	
	def __len__(self):
		lenx = len(self.x)
		leny = len(self.y)
		assert lenx == leny
		return lenx

from joblib import load

class MyLazyLoadDataset(torch.utils.data.Dataset):
	def __init__(self, data_root):
		self.root = data_root
		self.files = os.listdir(self.root)

		def extract_index(filename):
			return int(filename.split('_')[-1].split('.')[0])

		# 对文件列表进行排序，按照文件名中的序数进行排序
		sorted_file_list = sorted(self.files, key=extract_index)
		self.files = sorted_file_list


		self.file_length = len(self.files)
		self.dataset = None
		self.length_per_dataset = None
		self.length = None
		self.dataset_index = None

	def __getitem__(self, idx):
		start = idx.start
		stop = idx.stop
		index = stop // self.length_per_dataset
		start = idx.start % self.length_per_dataset
		# stop = idx.stop % self.length_per_dataset
		stop = start + idx.stop - idx.start
		if index >= len(self.files):
			index = self.dataset_index
		if index != self.dataset_index:
			file_name = os.path.join(self.root, self.files[index])  # load the features of this sample
			del self.dataset
			self.dataset = load(file_name)
			self.dataset_index = index
			data = self.dataset[start:stop]
			x = [i[0] for i in data]
			y = torch.tensor([i[1] for i in data])

			mydataset = MyDataset(x, y)
		else:
			data = self.dataset[start:stop]
			x = [i[0] for i in data]
			y = torch.tensor([i[1] for i in data])

			mydataset = MyDataset(x, y)

		return_dataset = mydataset[::]#[start:stop]
		return return_dataset

	def __len__(self):
		if self.length != None:
			return self.length
		sum_length = 0
		first_file_name = os.path.join(self.root, self.files[0])
		last_file_name = os.path.join(self.root, self.files[-1])

		first_features = load(first_file_name)
		y_length = len(first_features)
		if first_file_name == last_file_name:
			self.length_per_dataset = y_length
			self.length = y_length
			return self.length
		last_features = load(last_file_name)
		last_y_length = len(last_features)
		self.length_per_dataset = y_length
		sum_length = self.length_per_dataset * (len(self.files)-1) + last_y_length
		self.length = sum_length

		return sum_length


class MyParallelDataset(torch.utils.data.Dataset):
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def __getitem__(self, item):
		xitem = self.x[item]
		yitem = self.y[item]
		xitem = self.apply_to_device_and_convert(xitem, torch.device("cuda"))
		# xitem = xitem.to(torch.device("cuda"))
		yitem = yitem.to(torch.device("cuda"))

		return xitem, yitem

	def __len__(self):
		lenx = len(self.x)
		leny = len(self.y)
		assert lenx == leny
		return lenx

	def apply_to_device_and_convert(self, data, device):
		if isinstance(data, list):
			return [self.apply_to_device_and_convert(item, device) for item in data]
		elif isinstance(data, tuple):
			return tuple(self.apply_to_device_and_convert(list(data), device))
		elif isinstance(data, torch.Tensor):
			return data.to(device)
		elif isinstance(data, Data):
			data = data.clone()
			if hasattr(data, 'x') and isinstance(data.x, torch.Tensor):
				data.x = data.x.to(device)
			if hasattr(data, 'edge_index') and isinstance(data.edge_index, torch.Tensor):
				data.edge_index = data.edge_index.to(device)
			if hasattr(data, 'edge_attr') and isinstance(data.edge_attr, torch.Tensor):
				data.edge_attr = data.edge_attr.to(device)
			if hasattr(data, 'y') and isinstance(data.y, torch.Tensor):
				data.y = data.y.to(device)
			if hasattr(data, 'pos') and isinstance(data.pos, torch.Tensor):
				data.pos = data.pos.to(device)
			if hasattr(data, 'batch') and isinstance(data.batch, torch.Tensor):
				data.batch = data.batch.to(device)
			if hasattr(data, 'ptr') and isinstance(data.ptr, torch.Tensor):
				data.ptr = data.ptr.to(device)
			return data
		else:
			return data
class MyComplexDataset(torch.utils.data.Dataset):
	def __init__(self, x, y):
		self.x = x
		self.y = y
	
	def __getitem__(self, item):
		return self.x[item], self.y[item]
	
	def __len__(self):
		lenx = len(self.x)
		leny = len(self.y)
		assert lenx == leny
		return lenx


# %%
class MyDataLoader:
	def __init__(self, dataset, batch_size=1, shuffle=False):
		self.dataset = dataset
		self.batch_size = batch_size
		self.idx = 0
		self.dataset_len = len(self.dataset)
		
		# if shuffle:
		#     self.dataset = list(self.dataset)
		#     np.random.shuffle(self.dataset)
	
	def __getitem__(self, item):
		# 总长度
		with_batch_len = (self.dataset_len - 1) // self.batch_size + 1
		if isinstance(item, slice):
			# 是否是切片操作
			res = []
			start = item.start if item.start else 0
			stop = item.stop if item.stop else with_batch_len
			if start > with_batch_len or stop > with_batch_len or start < 0 or stop < 0:
				raise Exception('start >= with_batch_len or stop >= with_batch_len or start < 0 or stop < 0 ', start,
				                stop)
			for i in range(start, stop):
				res.append(self.dataset[i * self.batch_size:(i + 1) * self.batch_size])
			return res
		else:
			if item >= with_batch_len:
				raise Exception('item >= self.dataset_len')
			return self.dataset[item * self.batch_size:(item + 1) * self.batch_size]
	
	def __iter__(self):
		return self
	
	def __len__(self):
		return (self.dataset_len - 1) // self.batch_size + 1
	
	def __next__(self):
		if self.idx >= self.dataset_len:
			self.idx = 0
			raise StopIteration
		res = self.dataset[self.idx:self.idx + self.batch_size]
		self.idx += self.batch_size
		return res


class MyLazyDataLoader:
	def __init__(self, dataset, batch_size=1, shuffle=False):
		self.dataset = dataset
		self.batch_size = batch_size
		self.idx = 0
		self.dataset_len = len(self.dataset)

	def __getitem__(self, item):
		# 总长度
		with_batch_len = (self.dataset_len - 1) // self.batch_size + 1
		if isinstance(item, slice):
			# 是否是切片操作
			res = []
			start = item.start if item.start else 0
			stop = item.stop if item.stop else with_batch_len
			if start > with_batch_len or stop > with_batch_len or start < 0 or stop < 0:
				raise Exception('start >= with_batch_len or stop >= with_batch_len or start < 0 or stop < 0 ', start,
								stop)
			for i in range(start, stop):
				res.append(self.dataset[i * self.batch_size:(i + 1) * self.batch_size])
			return res
		else:
			if item >= with_batch_len:
				raise Exception('item >= self.dataset_len')
			return self.dataset[item * self.batch_size:(item + 1) * self.batch_size]

	def __iter__(self):
		return self

	def __len__(self):
		return (self.dataset_len - 1) // self.batch_size + 1

	def __next__(self):
		if self.idx >= self.dataset_len:
			self.idx = 0
			raise StopIteration
		res = self.dataset[self.idx:self.idx + self.batch_size]
		self.idx += self.batch_size
		return res


class MyComplexDataLoader:
	def __init__(self, dataset, batch_size=1, shuffle=False):
		self.dataset = dataset
		self.batch_size = batch_size
		self.idx = 0
		self.dataset_len = len(self.dataset)

	# if shuffle:
	#     self.dataset = list(self.dataset)
	#     np.random.shuffle(self.dataset)

	def __getitem__(self, item):
		# 总长度
		with_batch_len = (self.dataset_len - 1) // self.batch_size + 1
		if isinstance(item, slice):
			# 是否是切片操作
			res = []
			start = item.start if item.start else 0
			stop = item.stop if item.stop else with_batch_len
			if start > with_batch_len or stop > with_batch_len or start < 0 or stop < 0:
				raise Exception('start >= with_batch_len or stop >= with_batch_len or start < 0 or stop < 0 ', start,
								stop)
			for i in range(start, stop):
				res.append(self.dataset[i * self.batch_size:(i + 1) * self.batch_size])
			return res
		else:
			if item >= with_batch_len:
				raise Exception('item >= self.dataset_len')
			return self.dataset[item * self.batch_size:(item + 1) * self.batch_size]

	def __iter__(self):
		return self

	def __len__(self):
		return (self.dataset_len - 1) // self.batch_size + 1

	def __next__(self):
		if self.idx >= self.dataset_len:
			self.idx = 0
			raise StopIteration
		res = self.dataset[self.idx:self.idx + self.batch_size]
		self.idx += self.batch_size
		return res