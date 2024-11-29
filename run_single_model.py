import argparse
import pprint
import torch
import torch.utils.data.distributed
# from torch.utils.data.distributed import dis
import torch.distributed as dist
from torchgen.gen_lazy_tensor import default_args

# torch.autograd.set_detect_anomaly(True)

from transformers import *
import transformers as transformers
from transformers import (
	RobertaTokenizer,
	RobertaConfig,
)
import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '8888'

import sys
p = ("--batch-size 8 "
	 "--test-batch-size 8 "
	 "--epochs 4 " # 4
	 "--fine-tune-epochs 8 " # 8
	 "--lr 0.001 "
	 "--fine-tune-lr 0.000005 "
	 "--adam-epsilon 0.000001 "
	 "--max-seq-length 128 "
	 "--subtask-id A "
	 "--with-lm "
	 "--with-kegat "
	 "--with-kemb "
	 "--with-gnn "
	 "--with-plm "
	 "--with-scl " #ecnu comment
	 "--with-mps " #ecnu comment	 
	 "--with-gcnii " #ecnu comment
	 "--with-cache " #ecnu comment
	 "--latent-entities 128 " # ecnu comment
	 "--with-mult-write "
	 "--dataset arc_hard"
	 )
p = p.split(" ")
# sys.argv[1:] = p
pre_folder = "/data/chao_lei/ECNU/"
#pre_folder = "./"

from config import args as default_args, project_root_path
from model_modify import (
	create_datasets_with_kbert,
	create_lazy_load_datasets_with_kbert,
	create_parallel_datasets_with_kbert,
	train_and_finetune, test,
	train_and_finetune_and_test
)
from joblib import load, dump
import pickle
from model_modify_bp import (
load_checkpoint
)
from model_modify import (
test_with_dict
)

from models_bp import (
	SCL_SOTA_goal_model,
	SOTA_goal_model_plm_only,
	SOTA_goal_model_gnn_only,
	SCL_SOTA_goal_model_GNN_PLM_GCNII,
	RobertaForMultipleChoiceWithLM,
	RobertaForMultipleChoice, SOTA_goal_model_gat_gcnii_only
)
from utils.MyDataset import MyDataLoader, MyParallelDataset, MyLazyDataLoader
from utils.semevalUtils_bp import (
	get_all_features_from_task_1,
	get_all_features_from_task_2,
	get_all_complex_features_from_task_1,
	get_all_features_from_task_1_with_scl,
	get_all_features_from_task_1_with_scl_mps,
	get_all_features_from_task_1_with_mps_gcnii,
	get_all_features_from_task_1_with_scl_mps_gcnii,
)
# from accelerate import dispatch_model, infer_auto_device_map
# from accelerate.utils import get_balanced_memory

import time, sched

os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"

def build_parse():
	parser = argparse.ArgumentParser(description='ECNU-SenseMaker single model')
	parser.add_argument('--batch-size', type=int, default=default_args['batch_size'], metavar='N',
	                    help='input batch size for training (default: {})'.format(default_args['batch_size']))
	parser.add_argument('--test-batch-size', type=int, default=default_args['test_batch_size'], metavar='N',
	                    help='input batch size for testing (default: {})'.format(default_args['test_batch_size']))
	parser.add_argument('--epochs', type=int, default=default_args['epochs'], metavar='N',
	                    help='number of epochs to train (default: {})'.format(default_args['epochs']))
	parser.add_argument('--fine-tune-epochs', type=int, default=default_args['fine_tune_epochs'], metavar='N',
	                    help='number of fine-tune epochs to train (default: {})'.format(
		                    default_args['fine_tune_epochs']))
	parser.add_argument('--lr', type=float, default=default_args['lr'], metavar='LR',
	                    help='learning rate (default: {})'.format(default_args['lr']))
	parser.add_argument('--fine-tune-lr', type=float, default=default_args['fine_tune_lr'], metavar='LR',
	                    help='fine-tune learning rate (default: {})'.format(default_args['fine_tune_lr']))
	parser.add_argument('--adam-epsilon', type=float, default=default_args['adam_epsilon'], metavar='M',
	                    help='Adam epsilon (default: {})'.format(default_args['adam_epsilon']))
	parser.add_argument('--max-seq-length', type=int, default=default_args['max_seq_length'], metavar='N',
	                    help='max length of sentences (default: {})'.format(default_args['max_seq_length']))
	parser.add_argument('--subtask-id', type=str, default=default_args['subtask_id'],
	                    required=False, choices=['A', 'B'],
	                    help='subtask A or B (default: {})'.format(default_args['subtask_id']))
	parser.add_argument('--with-lm', action='store_true', default=False,
	                    help='Add Internal Sharing Mechanism (LM)')
	parser.add_argument('--with-scl', action='store_true', default=False,
	                    help='Add Supervised Contrastive Learning (SCL)')
	parser.add_argument('--with-complex', action='store_true', default=False,
	                    help='Add Complex Operation (Complex)')
	parser.add_argument('--with-mps', action='store_true', default=False,
	                    help='Add Maximum Path Sampling')
	parser.add_argument('--with-var', action='store_true', default=False,
	                    help='Add Variational Inference & Reparameter to Sampling')
	parser.add_argument('--with-kegat', action='store_true', default=False,
	                    help='Add Knowledge-enhanced Graph Attention Network (KEGAT)')
	parser.add_argument('--with-kemb', action='store_true', default=False,
	                    help='Add Knowledge-enhanced Embedding (KEmb)')
	parser.add_argument('--no-cuda', action='store_true', default=False,
	                    help='disables CUDA training')
	# parser.add_argument('--dry-run', action='store_true', default=False,
	#                     help='quickly check a single pass')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
	                    help='random seed (default: 1)')
	# parser.add_argument('--save-model', action='store_true', default=False,
	#                     help='For Saving the current Model')
	parser.add_argument('--with-cache', action='store_true', default=False, help='Add the serialized data obj')
	parser.add_argument('--with-gnn', action='store_true', default=False, help='Add the GNN')
	parser.add_argument('--use-multi-gpu', action='store_true', default=False, help='Apply multiple gpu')
	parser.add_argument('--with-plm', action='store_true', default=False, help='Add the PLM')
	parser.add_argument('--with-gcnii', action='store_true', default=False, help='Add the GCNII')
	parser.add_argument('--with-mult-write', action='store_true', default=False, help='Add the batch datasets load')
	parser.add_argument('--latent-entities', type=int, default=default_args['latent_entities'], metavar='N',
	                    help='latent entities number (default: {})'.format(default_args['lr']))
	parser.add_argument('--dataset', type=str, default=None, help='customized dataset', metavar='N')
	parser.add_argument('--knowledge-base', type=str, default='conceptnet', help='customized kb', metavar='N')

	args = parser.parse_args()  # 获取用户输入的参数

	torch.manual_seed(args.seed)

	for key in default_args.keys():
		# 将输入的参数更新至 default_args
		if hasattr(args, key):
			default_args[key] = getattr(args, key)
	default_args['use_cuda'] = not args.no_cuda and torch.cuda.is_available()

	default_args['device'] = torch.device('cuda' if default_args['use_cuda'] else 'cpu')
	default_args['with_lm'] = args.with_lm
	default_args['with_kegat'] = args.with_kegat
	default_args['with_kemb'] = args.with_kemb
	default_args['with_scl'] = args.with_scl
	default_args['with_complex'] = args.with_complex
	default_args['with_mps'] = args.with_mps
	default_args['with_cache'] = args.with_cache
	default_args['with_gnn'] = args.with_gnn
	default_args['with_plm'] = args.with_plm
	default_args['with_gcnii'] = args.with_gcnii
	default_args['with_mps'] = args.with_mps
	default_args['dataset'] = args.dataset
	default_args['knowledge_base'] = args.knowledge_base
	default_args['use_multi_gpu'] = args.use_multi_gpu
	default_args['with_mult_write'] = args.with_mult_write

	return args


if __name__ == '__main__':
	build_parse()
		# print all config
	print(project_root_path)
	pprint.pprint(default_args)
	
	# prepare for tokenizer and model
	tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
	config = RobertaConfig.from_pretrained('roberta-large')
	
	config.hidden_dropout_prob = 0.2
	config.attention_probs_dropout_prob = 0.2


	if default_args['with_gnn']:
		if default_args['with_plm']:
			if default_args['with_gcnii']:
				model = SCL_SOTA_goal_model_GNN_PLM_GCNII(default_args)
			else:
				model = SCL_SOTA_goal_model(default_args)
		elif default_args['with_gcnii']:
			model = SOTA_goal_model_gat_gcnii_only(default_args)
		else:
			model = SOTA_goal_model_gnn_only(default_args)
	else:
		if default_args['with_plm']:
			model = SOTA_goal_model_plm_only(default_args)
		else:
			assert False
	print(model)
	"""
	if default_args['with_kegat']:
		# create a model with kegat (or and lm)
		model = SCL_SOTA_goal_model(default_args)
	elif default_args['with_lm']:
		model = RobertaForMultipleChoiceWithLM.from_pretrained(
			'pre_weights/roberta-large_model.bin', config=config)
	else:
		model = RobertaForMultipleChoice.from_pretrained(
			'pre_weights/roberta-large_model.bin', config=config)
	"""
	train_data = test_data = optimizer = None
	
	print('with_LM: ', 'Yes' if default_args['with_lm'] else 'No')
	print('with_KEGAT: ', 'Yes' if default_args['with_kegat'] else 'No')
	print('with_KEmb: ', 'Yes' if default_args['with_kemb'] else 'No')

	datasets = ['ai2_science_elementary', 'ai2_science_middle', 'arc_hard', 'arc_easy']
	# dataset_dir = 'arc_hard'
	assert default_args['dataset'] != None
	dataset_dir = default_args['dataset']

	if not default_args['with_cache']:
		train_features, dev_features, test_features = [], [], []

		data_prepare_function = None
		if default_args['with_scl']:
			if default_args['with_mps']:
				data_prepare_function = get_all_features_from_task_1_with_scl_mps
				if default_args['with_gcnii']:
					# NotImplementedError("SCL MPS GCNII not implemented!")
					data_prepare_function = get_all_features_from_task_1_with_mps_gcnii
					# temporarily implemented by mps gcnii
			else:
				data_prepare_function = get_all_features_from_task_1_with_scl
		elif default_args['with_complex']:
			data_prepare_function = get_all_complex_features_from_task_1
		elif default_args['with_mps']:
			if default_args['with_plm']:
				NotImplementedError("PLM MPS not implemented!")
			elif default_args['with_gcnii']:
					data_prepare_function = get_all_features_from_task_1_with_mps_gcnii
		else:
			data_prepare_function = get_all_features_from_task_1 # this is the ecnu implementation
		# train_features = get_all_features_from_task_1(
		# SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Testing Data/

		print("Processing the dataset:" + dataset_dir)
		train_features = data_prepare_function(
			# train_features = get_all_complex_features_from_task_1(
			# 'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Training Data/subtaskA_data_all.csv',
			dataset_dir + "/train/data_all.csv",
			# 'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Training Data/subtaskA_data_all_pure.csv',
			# 'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Training Data/subtaskA_answers_all.csv',
			dataset_dir + "/train/answers_all.csv",
			# 'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Training Data/subtaskA_answers_all_pure.csv',
			tokenizer, default_args['max_seq_length'],
			with_gnn=default_args['with_kegat'],
			with_k_bert=default_args['with_kemb'],
			# argument=True,
			latent_entities=default_args['latent_entities'],
			batch_size=default_args['batch_size']
		)

		dev_features = data_prepare_function(
			# dev_features = get_all_complex_features_from_task_1(
			# 'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Dev Data/subtaskA_dev_data.csv',
			# 'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Dev Data/subtaskA_dev_data_pure.csv',
			dataset_dir + "/dev/data_all.csv",
			# 'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Dev Data/subtaskA_gold_answers.csv',
			# 'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Dev Data/subtaskA_gold_answers_pure.csv',
			dataset_dir + "/dev/answers_all.csv",
			tokenizer, default_args['max_seq_length'],
			with_gnn=default_args['with_kegat'],
			with_k_bert=default_args['with_kemb'],
			# argument =True,
			batch_size=default_args['batch_size'],
			latent_entities=default_args['latent_entities'],
		)

		test_features = data_prepare_function(
			# 'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Testing Data/subtaskA_test_data_pure.csv',
			dataset_dir + "/test/data_all.csv",
			# 'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Testing Data/subtaskA_gold_answers_pure.csv',
			dataset_dir + "/test/answers_all.csv",
			tokenizer, default_args['max_seq_length'],
			with_gnn=default_args['with_kegat'],
			with_k_bert=default_args['with_kemb'],
			# argument= True,
			batch_size=default_args['batch_size'],
			latent_entities=default_args['latent_entities'],
		)

		"""
		if default_args['with_gnn']:
			if default_args['with_plm']:
				if default_args['with_gcnii']:
					if default_args['with_scl']:
						pass
					elif default_args['with_complex']:
						pass
					elif default_args['with_mps']:
						pass
					else:
						# This is sota_goal_model setting
						print("dumping the train dataset")
						dump(train_features, "processed/" + dataset_dir + "/train/train_all.joblib")
						print("dumping the dev dataset")
						dump(dev_features, "processed/" + dataset_dir + "/dev/dev_all.joblib")
						print("dumping the test dataset")
						dump(test_features, "processed/" + dataset_dir + "/test/test_all.joblib")
		"""

		train_dataset = create_datasets_with_kbert(train_features, shuffle=False)
		dev_dataset = create_datasets_with_kbert(dev_features, shuffle=False)
		test_dataset = create_datasets_with_kbert(test_features, shuffle=False)

		train_data = MyDataLoader(train_dataset, batch_size=default_args['batch_size'])
		dev_data = MyDataLoader(dev_dataset, batch_size=default_args['test_batch_size'])
		test_data = MyDataLoader(test_dataset, batch_size=default_args['test_batch_size'])
	else:
		print("loading the cache data:" + dataset_dir)
		if default_args['with_mult_write']:
			train_features = pre_folder + "processed/" + dataset_dir + "/train/" # train_all.joblib"
			dev_features = pre_folder + "processed/" + dataset_dir + "/dev/"  # dev_all.joblib"
			test_features = pre_folder + "processed/" + dataset_dir + "/test/" #test_all.joblib"
			# train_features = test_features
			# dev_features = test_features
		elif default_args['with_gcnii']:
			from itertools import chain
			# train_features_0 = load("processed_dataset/train/gcnii_pure_train_0.joblib")
			# train_features_1 = load("processed_dataset/train/gcnii_pure_train_1.joblib")
			# merged_features = chain(train_features_0, train_features_1)
			# train_features = list(merged_features)

			train_features = load(pre_folder + "processed/" + dataset_dir + "/train/train_all.joblib")
			# dev_features = train_features[0:1000*default_args['batch_size']]
			# test_features = dev_features
			# dev_features = load(pre_folder + "processed/" + dataset_dir + "/test/test_all.joblib")
			dev_features = load(pre_folder + "processed/" + dataset_dir + "/dev/dev_all.joblib")
			# train_features = dev_features
			# test_features = dev_features
			test_features = load(pre_folder + "processed/" + dataset_dir + "/test/test_all.joblib")

			# dev_features = test_features
			# test_features = dev_features
		else:
			with open('serialized_train_data.pkl', 'rb') as file:
				train_data = pickle.loads(file.read())
			with open('serialized_dev_data.pkl', 'rb') as file:
				dev_data = pickle.loads(file.read())
			with open('serialized_test_data.pkl', 'rb') as file:
				test_data = pickle.loads(file.read())

		if default_args['with_mult_write']:
			train_dataset = create_lazy_load_datasets_with_kbert(train_features, shuffle=False)
			dev_dataset = create_lazy_load_datasets_with_kbert(dev_features, shuffle=False)
			test_dataset = create_lazy_load_datasets_with_kbert(test_features, shuffle=False)
		else:
			train_dataset = create_datasets_with_kbert(train_features, shuffle=False)
			dev_dataset = create_datasets_with_kbert(dev_features, shuffle=False)
			test_dataset = create_datasets_with_kbert(test_features, shuffle=False)


		if default_args['use_multi_gpu']:
			dist.init_process_group(backend='nccl', init_method='env://', rank=torch.cuda.device_count(), world_size=1)
			train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
			train_data = torch.utils.data.DataLoader(train_dataset, batch_size=default_args['batch_size'],
													 sampler=train_sampler)
			dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_dataset)
			dev_data = torch.utils.data.DataLoader(dev_dataset, batch_size=default_args['test_batch_size'],
												   sampler=dev_sampler)
			test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
			test_data = torch.utils.data.DataLoader(test_dataset, batch_size=default_args['test_batch_size'],
													sampler=test_sampler)
		elif default_args['with_mult_write']:
			train_data = MyLazyDataLoader(train_dataset, batch_size=default_args['batch_size'])
			dev_data = MyLazyDataLoader(dev_dataset, batch_size=default_args['test_batch_size'])
			test_data = MyLazyDataLoader(test_dataset, batch_size=default_args['test_batch_size'])
		else:
			train_data = MyDataLoader(train_dataset, batch_size=default_args['batch_size'])
			dev_data = MyDataLoader(dev_dataset, batch_size=default_args['test_batch_size'])
			test_data = MyDataLoader(test_dataset, batch_size=default_args['test_batch_size'])

			train_data = list(train_data)
			dev_data = list(dev_data)
			test_data = list(test_data)

	print('train_data len: ', len(train_data))
	print('dev_data len: ', len(dev_data))
	print('test_data len: ', len(test_data))

	# dev_acc, (train_pred_opt, dev_pred_opt) = train_and_finetune_and_test(model, train_data, dev_data, test_data, default_args)
	dev_acc, (train_pred_opt, dev_pred_opt) = train_and_finetune(model, train_data, dev_data, default_args)
	loss, acc, pred, mydict = test_with_dict(model, test_data, default_args)

	"""
	best_model_temp_path = os.path.join("./models/best_scl.pth")
	if os.path.isfile(best_model_temp_path):
		model.load_state_dict(torch.load(
			best_model_temp_path
		))
	"""

	# _, test_acc, _ = test(model, test_data, default_args)
	print('Dev acc: ', dev_acc)
	print(mydict)
	# print('Test acc: ', test_acc)