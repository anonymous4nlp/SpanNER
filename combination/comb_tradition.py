# -*- coding: utf-8 -*
import codecs
import numpy as np
from collections import Counter
import os
import pickle
import random
from evaluate_metric import get_chunks, get_chunks_onesent,evaluate_chunk_level,evaluate_each_class,evaluate_ByCategory

# traditional baseline
import codecs
import argparse
import numpy as np
from collections import Counter
import os
import pickle
import random
from evaluate_metric import get_chunks, get_chunks_onesent,evaluate_chunk_level,evaluate_each_class
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn import svm

from dataread import DataReader
from copy import deepcopy

# def get_glove_emb(fn_glove,spans):


class CombTraditional():
	def __init__(self, dataname, file_dir, useSpanLen, useMorph,useOverallF1):
		self.dataname = dataname
		self.file_dir = file_dir
		self.useMorph = useMorph
		self.useSpanLen = useSpanLen
		self.useOverallF1 =useOverallF1
		self.maxSpanLen = 10



	def get_datalabel2idx(self,corpus_type):
		label = []
		if 'conll' in corpus_type:
			label = ["O", "ORG", "PER", "LOC", "MISC"]
		elif 'note' in corpus_type:
			label = ['O', 'PERSON', 'ORG', 'GPE', 'DATE', 'NORP', 'CARDINAL', 'TIME',
					 'LOC', 'FAC', 'PRODUCT', 'WORK_OF_ART', 'MONEY', 'ORDINAL',
					 'QUANTITY', 'EVENT', 'PERCENT', 'LAW', 'LANGUAGE']
		elif corpus_type=='wnut16':
			label = ['O','loc', "facility", "movie", "company", "product", "person", "other", "tvshow", "musicartist",
					   "sportsteam"]
		elif corpus_type == 'wnut17':
			label = ['O',"location", "group", "corporation", "person", "creative-work", "product"]
		else:
			print("error, unrecognize the corpus_type!!!!!")

		label2idx = {}
		for idx, x in enumerate(label):
			label2idx[x] = idx

		return label2idx


	def get_unique_pchunk_labs(self,dataname, file_dir, classes, fmodels_dev, fn_stand_res_dev):
		mres = DataReader(dataname, file_dir, classes, fmodels_dev, fn_stand_res_dev)

		tchunks_models, \
		tchunks_unique, \
		pchunks_models, \
		tchunks_models_onedim, \
		pchunks_models_onedim, \
		pchunk2label_models, \
		tchunk2label_dic, \
		class2f1_models = mres.get_allModels_pred()

		# get words
		wordseq_sent = mres.get_sent_word()

		# the unique chunk that predict by the model..
		pchunks_unique= list(set(pchunks_models_onedim))

		# get the unique non-O chunk's label that are predicted by all the 10 models.
		keep_pref_upchunks = []
		pchunk_plb_ms =[]
		for pchunk in pchunks_unique:
			lab, sid, eid, sentid = pchunk
			key1 = (sid, eid, sentid)
			if key1 not in keep_pref_upchunks:
				keep_pref_upchunks.append(key1)
				plb_ms = [] # the length is the num of the models
				# plb_ms.append(pchunk) # the first position is the pchunk
				for i in range(len(fmodels_dev)):
					plb = 'O'
					if key1 in pchunk2label_models[i]:
						plb = pchunk2label_models[i][key1]
					plb_ms.append(plb)
				pchunk_plb_ms.append(plb_ms)

		# get the non-O true chunk that are not be recognized..
		for tchunk in tchunks_unique:
			if tchunk not in pchunks_unique: # it means that the tchunk are not been recognized by all the models
				plab, sid, eid, sentid = tchunk
				key1 = (sid, eid, sentid)
				if key1 not in keep_pref_upchunks:
					plb_ms = ['O' for i in range(len(fmodels_dev))]
					pchunk_plb_ms.append(plb_ms)
					keep_pref_upchunks.append(key1)

		return pchunk_plb_ms,keep_pref_upchunks,tchunk2label_dic,wordseq_sent


	def get_spanMorph(self,wordseq,pref_chunk):
		morph2idx = {'isupper': 1, 'islower': 2, 'istitle': 3, 'isdigit': 4, 'other': 5}
		sid, eid, sentid = pref_chunk
		span = wordseq[sid:eid]
		# print('span: ',span)

		caseidxs = []

		for j, token in enumerate(span):
			if token.isupper():
				tfeat = 'isupper'
			elif token.islower():
				tfeat = 'islower'
			elif token.istitle():
				tfeat = 'istitle'
			elif token.isdigit():
				tfeat = 'isdigit'
			else:
				tfeat = 'other'
			caseidxs.append(morph2idx[tfeat])

		if len(caseidxs)<self.maxSpanLen:
			padLen = self.maxSpanLen-len(caseidxs)
			for i in range(padLen):
				caseidxs.append(0)

		return caseidxs



	# get the level1's input data.
	def get_train_data(self,dataname, file_dir, classes, fmodels_dev, fn_stand_res_dev,cf1):
		pchunk_plb_ms, keep_pref_upchunks,tchunk2label_dic,wordseq_sent = self.get_unique_pchunk_labs(dataname, file_dir, classes, fmodels_dev, fn_stand_res_dev)
		print("len(pchunk_plb_ms): ", len(pchunk_plb_ms))
		print("len(keep_upchunks): ", len(keep_pref_upchunks))
		assert len(pchunk_plb_ms) == len(keep_pref_upchunks)



		# print('self.label2idx: ',label2idx)
		self.label2idx = self.get_datalabel2idx(self.dataname)

		comb_kchunks = []
		X_train = []
		Y_train = []
		span_len =0
		print('self.useOverallF1,self.useSpanLen,self.useMorph')
		print(self.useOverallF1,self.useSpanLen,self.useMorph)
		for pchunk_plb_m, pref_upchunks in zip(pchunk_plb_ms, keep_pref_upchunks):
			lb2num_dic = {}
			plabs = []
			sid, eid, sentid = pref_upchunks
			span_len = eid-sid
			# print('span_len: ',span_len)

			for plbm in pchunk_plb_m:
				# print('plbm: ',plbm)
				lidx = self.label2idx[plbm]
				plabs.append(lidx)

			# add the morph feat
			if self.useOverallF1:
				plabs+=cf1
			if self.useSpanLen:
				plabs.append(span_len)
			if self.useMorph:
				wordseq = wordseq_sent[sentid]
				caseidxs = self.get_spanMorph(wordseq, pref_upchunks)
				plabs +=caseidxs



				# print('plabs: ',plabs)
			plabs = [float(lab) for lab in plabs]
			X_train.append(plabs)
			# print('plabs: ',plabs)


			# save the true label for the current span.
			tlab = 'O'
			if pref_upchunks in tchunk2label_dic:
				tlab = tchunk2label_dic[pref_upchunks]
			tlab_idx = self.label2idx[tlab]
			Y_train.append(tlab_idx)


		X_train = np.array(X_train)
		Y_train = np.array(Y_train)
		# print('Y_train: ', Y_train)

		print('sum(Y_train!=0)',sum(Y_train!=0))
		return X_train,Y_train

	def get_tradCombinator(self,dataname, file_dir, classes, fmodels_dev, fn_stand_res_dev,fmodels_test, fn_stand_res_test,cf1):
		X_train, Y_train = self.get_train_data(dataname, file_dir, classes, fmodels_dev, fn_stand_res_dev,cf1)
		X_test, Y_test = self.get_train_data(dataname, file_dir, classes, fmodels_test, fn_stand_res_test,cf1)

		# mlp_res,mlp_res_byprob = self.combinator_MLPClassifier(X_train, Y_train, X_test, Y_test)
		svm_res,svm_res_byprob,predict_svm = self.combinator_SVM(X_train, Y_train, X_test, Y_test)
		rfc_res, rfc_res_byprob,predict_rfc = self.combinator_RandomForestClassifier(X_train, Y_train, X_test, Y_test)
		xgb_res,xgb_res_byprob,predict_xgb = self.combinator_XGBClassifier(X_train, Y_train, X_test, Y_test)

		# dtc_res,dtc_res_byprob = self.combinator_DecisionTreeClassifier(X_train, Y_train, X_test, Y_test)
		# kgc_res,kgc_res_byprob = self.combinator_KNeighborsClassifier(X_train, Y_train, X_test, Y_test)
		# gnb_res,gnb_res_byprob = self.combinator_GaussianNB(X_train, Y_train, X_test, Y_test)
		# lr_res,lr_res_byprob = self.combinator_LogisticRegression(X_train, Y_train, X_test, Y_test)
		# mnb_res,comb_res_byprob = self.combinator_MultinomialNB(X_train, Y_train, X_test, Y_test)

		res_normal = [svm_res[0],rfc_res[0],xgb_res[0]]
		res_prob = [svm_res_byprob[0],rfc_res_byprob[0],xgb_res_byprob[0]]
		print('res_prob: ',res_prob)

		self.save_comb_res(predict_svm, Y_test, dataname, file_dir, classes, fmodels_test, fn_stand_res_test,'SVM',svm_res_byprob[0])
		self.save_comb_res(predict_rfc, Y_test, dataname, file_dir, classes, fmodels_test, fn_stand_res_test, 'RFC',rfc_res_byprob[0])
		self.save_comb_res(predict_xgb, Y_test, dataname, file_dir, classes, fmodels_test, fn_stand_res_test, 'XGB',xgb_res_byprob[0])

		# return [svm_res[0],rfc_res[0],xgb_res[0],mlp_res[0],dtc_res[0],kgc_res[0],gnb_res[0],lr_res[0]]
		return {'res_normal':res_normal, 'res_prob':res_prob}

	def save_comb_res(self,predict,target,dataname, file_dir, classes, fmodels_test, fn_stand_res_test,model_name,f1):
		mres = DataReader(dataname, file_dir, classes, fmodels_test, fn_stand_res_test)
		pchunk_plb_ms, keep_pref_upchunks, tchunk2label_dic, wordseq_sent = self.get_unique_pchunk_labs(dataname,
																										file_dir,
																										classes,
																										fmodels_test,
																										fn_stand_res_test)
		#index to label.
		idx2label = {}
		for lab, idx in self.label2idx.items():
			idx2label[idx] = lab


		print(len(keep_pref_upchunks))
		print(len(target))
		print(len(predict))
		pchunks = []
		tchunks = []

		for pre_chunk, tid, pid in zip(keep_pref_upchunks,target,predict):
			sid, eid, sentid = pre_chunk
			tlab = idx2label[tid]
			plab = idx2label[pid]
			if tid !=0:
				tchunk = (tlab,sid, eid, sentid)
				tchunks.append(tchunk)
			if pid !=0:
				pchunk = (plab,sid, eid, sentid)
				pchunks.append(pchunk)


		print('len(tchunks): ',len(tchunks))
		print('len(pchunks): ', len(pchunks))
		print(f1)
		kf1 = int(float(f1) * 100)
		fn_save_comb_kchunks = 'comb_result/'+model_name+'_6seq_2span_res' + str(kf1) + '.pkl'

		pickle.dump([pchunks, tchunks], open(fn_save_comb_kchunks, "wb"))


		# self.label2idx

	def compute_prf1_onlyPred(self,pred_prob, target,x_test):
		# this evaluation only consider the tag that have been predicted.
		# the predict is prob
		print("num target: ", len(target))
		print('num predict: ', len(pred_prob))
		print("sum(target!=0)", sum(target != 0))

		predict = []
		for x1,prob1 in zip(x_test,pred_prob):
			# print('prob1: ',prob1)

			ux1 = list(set(x1))
			# prob1 = list(prob1)
			if len(ux1)==1:
				predict.append(ux1[0])
			else:
				# print('ux1: ',ux1)
				pred_probs = []
				for idx in ux1:
					if idx <len(prob1):
						pred_probs.append(prob1[int(idx)])

				# pred_probs = [prob1[int(idx)] for idx in ux1]
				midx = pred_probs.index(max(pred_probs))
				klab_id = ux1[midx]
				predict.append(klab_id)
				# print('x_test1: ',x1)
				# print('prob1: ', prob1)
				# print('klab_id: ', klab_id)



		predict, target = np.array(predict), np.array(target)
		# correct_preds = sum(predict ==target & predict!=0)
		total_preds = sum(predict != 0)
		total_correct = sum(target != 0)

		correct_preds = 0
		for p, t in zip(predict, target):
			if p == t and p != 0:
				correct_preds += 1

		p = correct_preds / total_preds if correct_preds > 0 else 0
		r = correct_preds / total_correct if correct_preds > 0 else 0
		f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0

		cp = correct_preds
		tp = total_preds

		f1 = '%.2f' %(f1*100)
		p = '%.2f' %(p*100)
		r = '%.2f' %(r*100)

		print(f1, p, r, correct_preds, total_preds, total_correct)

		# self.save_comb_res(predict, target, dataname, file_dir, classes, fmodels_test, fn_stand_res_test)

		return [f1, p, r, correct_preds, total_preds, total_correct],predict


	def compute_prf1(self,predict, target):
		# predict, target are one dim.
		print("num target: ", len(target))
		print('num predict: ', len(predict))
		print("sum(target!=0)", sum(target != 0))
		predict, target = np.array(predict), np.array(target)
		# correct_preds = sum(predict ==target & predict!=0)
		total_preds = sum(predict != 0)
		total_correct = sum(target != 0)

		correct_preds = 0
		for p, t in zip(predict, target):
			if p == t and p != 0:
				correct_preds += 1

		p = correct_preds / total_preds if correct_preds > 0 else 0
		r = correct_preds / total_correct if correct_preds > 0 else 0
		f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0

		cp = correct_preds
		tp = total_preds

		f1 = '%.2f' %(f1*100)
		p = '%.2f' %(p*100)
		r = '%.2f' %(r*100)

		print(f1, p, r, correct_preds, total_preds, total_correct)
		return f1, p, r, correct_preds, total_preds, total_correct

	def combinator_MLPClassifier(self,x_train, y_train, x_test, y_test):
		clf = MLPClassifier(random_state=1, max_iter=300)
		clf.fit(x_train, y_train)
		predictions = clf.predict(x_test)
		comb_res = self.compute_prf1(predictions, y_test)
		print('MLPClassifier combinate F1: ', comb_res)

		pred_prob = clf.predict_proba(x_test)
		comb_res_byprob,predict = self.compute_prf1_onlyPred(pred_prob, y_test, x_test)
		print('MLPClassifier combinate F1 by prob: ', comb_res_byprob)

		return comb_res,comb_res_byprob,predict

	def combinator_SVM(self,x_train, y_train, x_test, y_test):
		clf = svm.SVC(probability=True)
		clf.fit(x_train, y_train)
		predictions = clf.predict(x_test)
		comb_res = self.compute_prf1(predictions, y_test)
		print('SVM combinate F1: ', comb_res)

		pred_prob = clf.predict_proba(x_test)
		comb_res_byprob, predict = self.compute_prf1_onlyPred(pred_prob, y_test, x_test)
		print('MLPClassifier combinate F1 by prob: ', comb_res_byprob)

		return comb_res, comb_res_byprob, predict

		# return comb_res, comb_res_byprob

	def combinator_XGBClassifier(self,x_train, y_train, x_test, y_test):
		# gbm = xgb.XGBClassifier(
		# 	# learning_rate = 0.02,
		# 	n_estimators=200,  # 2000,
		# 	max_depth=4,
		# 	min_child_weight=2,
		# 	# gamma=1,
		# 	gamma=0.9,
		# 	subsample=0.8,
		# 	colsample_bytree=0.8,
		# 	# objective='binary:logistic',
		# 	nthread=-1,
		# 	scale_pos_weight=1).fit(x_train, y_train)

		# gbm = xgb.XGBClassifier(max_depth=5, objective='multi:softprob', n_estimators=1000,
        #                 num_classes=5).fit(x_train, y_train)

		gbm = xgb.XGBClassifier(max_depth=5, objective='multi:softmax', n_estimators=100,
								num_classes=5).fit(x_train, y_train)

		predictions = gbm.predict(x_test)

		comb_res = self.compute_prf1(predictions, y_test)
		print('XGBClassifier combinate F1: ', comb_res)

		pred_prob = gbm.predict_proba(x_test)
		comb_res_byprob, predict = self.compute_prf1_onlyPred(pred_prob, y_test, x_test)
		print('MLPClassifier combinate F1 by prob: ', comb_res_byprob)

		return comb_res, comb_res_byprob, predict

		# return comb_res, comb_res_byprob

	def combinator_RandomForestClassifier(self,x_train, y_train, x_test, y_test):
		clf = RandomForestClassifier(random_state=1)
		clf.fit(x_train, y_train)
		predictions = clf.predict(x_test)

		comb_res = self.compute_prf1(predictions, y_test)
		print('RandomForestClassifier combinate F1: ', comb_res)

		pred_prob = clf.predict_proba(x_test)
		comb_res_byprob, predict = self.compute_prf1_onlyPred(pred_prob, y_test, x_test)
		print('MLPClassifier combinate F1 by prob: ', comb_res_byprob)

		return comb_res, comb_res_byprob, predict

	def combinator_LogisticRegression(self,x_train, y_train, x_test, y_test):
		clf = LogisticRegression()
		clf.fit(x_train, y_train)
		predictions = clf.predict(x_test)

		comb_res = self.compute_prf1(predictions, y_test)
		print('LogisticRegression combinate F1: ', comb_res)

		pred_prob = clf.predict_proba(x_test)
		comb_res_byprob = self.compute_prf1_onlyPred(pred_prob, y_test, x_test)
		print('LogisticRegression combinate F1 by prob: ', comb_res_byprob)

		return comb_res, comb_res_byprob

	def combinator_KNeighborsClassifier(self,x_train, y_train, x_test, y_test):
		clf = KNeighborsClassifier(n_neighbors=1)
		clf.fit(x_train, y_train)
		predictions = clf.predict(x_test)

		comb_res = self.compute_prf1(predictions, y_test)
		print('KNeighborsClassifier combinate F1: ', comb_res)

		pred_prob = clf.predict_proba(x_test)
		comb_res_byprob = self.compute_prf1_onlyPred(pred_prob, y_test, x_test)
		print('KNeighborsClassifier combinate F1 by prob: ', comb_res_byprob)

		return comb_res, comb_res_byprob

	def combinator_DecisionTreeClassifier(self,x_train, y_train, x_test, y_test):
		clf = DecisionTreeClassifier()
		clf.fit(x_train, y_train)
		predictions = clf.predict(x_test)

		comb_res = self.compute_prf1(predictions, y_test)
		print('DecisionTreeClassifier combinate F1: ', comb_res)

		pred_prob = clf.predict_proba(x_test)
		comb_res_byprob = self.compute_prf1_onlyPred(pred_prob, y_test, x_test)
		print('DecisionTreeClassifier combinate F1 by prob: ', comb_res_byprob)

		return comb_res, comb_res_byprob

	def combinator_GaussianNB(self,x_train, y_train, x_test, y_test):
		clf = GaussianNB()
		clf.fit(x_train, y_train)
		predictions = clf.predict(x_test)

		comb_res = self.compute_prf1(predictions, y_test)
		print('GaussianNB combinate F1: ', comb_res)

		pred_prob = clf.predict_proba(x_test)
		comb_res_byprob = self.compute_prf1_onlyPred(pred_prob, y_test, x_test)
		print('GaussianNB combinate F1 by prob: ', comb_res_byprob)

		return comb_res, comb_res_byprob

	def combinator_MultinomialNB(self,x_train, y_train, x_test, y_test):
		clf = MultinomialNB()
		clf.fit(x_train, y_train)
		predictions = clf.predict(x_test)

		comb_res = self.compute_prf1(predictions, y_test)
		print('MultinomialNB combinate F1: ', comb_res)

		pred_prob = clf.predict_proba(x_test)
		comb_res_byprob = self.compute_prf1_onlyPred(pred_prob, y_test, x_test)
		print('MultinomialNB combinate F1 by prob: ', comb_res_byprob)

		return comb_res, comb_res_byprob



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Training")
	parser.add_argument('--useSpanLen', type=int,default=0, nargs='?',
						help='True to use glove embeddings')
	parser.add_argument('--useMorph', type=int,default=0, nargs='?',
						help='True to use glove embeddings')
	parser.add_argument('--useOverallF1', type=int, default=0, nargs='?',
						help='True to use glove embeddings')

	args = parser.parse_args()

	corpus_types = ["notenw","conll03","notebn","notebc","notemz","notewb", "notetc","conll02dutch","conll02spanish","wnut16","wnut17"] #"conll03"
	# # corpus_types = ["conll02_spanish"]
	# # corpus_types = ["notebc"] "conll02spanish", "conll02dutch",
	# corpus_types = ["conll03",  "notenw", "notebn", "notebc", "notemz", "notewb", "notetc"]
	# # corpus_types = ["notemz", "notewb", "notetc"]
	# corpus_types = ["conll02spanish","wnut16","wnut17"]
	# corpus_types = ["notenw", "notebc", "notemz", "notewb", "notetc", ]
	# corpus_types =	["conll02dutch","conll02spanish", "wnut16", "wnut17"]  # "conll03"
	# corpus_types = ["wnut16", "wnut17"]
	corpus_types = ["conll03"]

	comb_results = {}
	column_true_tag_test, column_pred_tag_test = 1, 2
	classes = []
	fn_res = []
	file_dir =''
	fn_stand_res_test = ''
	fn_stand_res_dev = ''
	for corpus_type in corpus_types:
		if corpus_type =="conll03":
			fn_res =[
					"conll03_CflairWnon_lstmCrf_1_test_9241.txt",
					"conll03_CflairWnon_lstmCrf_1_dev.txt",
					"conll03_CbertWglove_lstmCrf_1_test_9201.txt",
					"conll03_CbertWglove_lstmCrf_1_dev.txt",
					"conll03_CbertWnon_lstmCrf_1_test_9246.txt",
					"conll03_CbertWnon_lstmCrf_1_dev.txt",
					"conll03_CflairWglove_lstmCrf_1_test_9302.txt",
					"conll03_CflairWglove_lstmCrf_1_dev.txt",
					"conll03_CelmoWglove_lstmCrf_95803618_test_9211.txt",
					"conll03_CelmoWglove_lstmCrf_95803618_dev.txt",
					"conll03_CcnnWglove_lstmCrf_72102467_test_9088.txt",
					"conll03_CcnnWglove_lstmCrf_72102467_dev.txt",
					"conll03_CcnnWglove_cnnCrf_45725566_test_8971.txt",
					"conll03_CcnnWglove_cnnCrf_45725566_dev.txt",
					"conll03_CelmoWnon_lstmCrf_81319158_test_9199.txt",
					"conll03_CelmoWnon_lstmCrf_81319158_dev.txt",
					"conll03_CnonWrand_lstmCrf_03689925_test_7849.txt",
					"conll03_CnonWrand_lstmCrf_03689925_dev.txt",
					"conll03_CcnnWrand_lstmCrf_43667285_test_8303.txt",
					"conll03_CcnnWrand_lstmCrf_43667285_dev.txt",
					"conll03_spanPred_test_9245.txt",
					"conll03_spanPred_dev.txt",
					"spanPred_prunFalse_spLenFalse_test_9157.txt",
					"spanPred_prunFalse_spLenFalse_dev.txt",
			]
			fn_stand_res_test = 'conll03_CcnnWglove_lstmCrf_72102467_test_9088.txt'
			fn_stand_res_dev = 'conll03_CcnnWglove_lstmCrf_72102467_dev.txt'
			fn_prob = "spanPred_bert-large-uncased_prunTrue_spLenTrue_spMorphFalse_SpWtFalse_value1_prob_test_9245.pkl"
			classes = ["ORG", "PER", "LOC", "MISC"]
		elif corpus_type == "notenw":
			fn_prob = "spanPred_bert-large-uncased_prunTrue_spLenTrue_spMorphFalse_SpWtFalse_value1_prob_test_9159.pkl"
			classes = ['PERSON', 'ORG', 'GPE', 'DATE', 'NORP', 'CARDINAL', 'TIME',
                     'LOC', 'FAC', 'PRODUCT', 'WORK_OF_ART', 'MONEY', 'ORDINAL',
                     'QUANTITY','EVENT', 'PERCENT', 'LAW', 'LANGUAGE']
			fn_stand_res_test = "notenw_CnonWrand_lstmCrf_62726359_test_8573.txt"
			fn_stand_res_dev = "notenw_CnonWrand_lstmCrf_62726359_dev.txt"
			fn_res= [
				"notenw_CelmoWglove_lstmCrf_64679144_test_9010.txt",
				"notenw_CelmoWglove_lstmCrf_64679144_dev.txt",
				"notenw_CflairWglove_lstmCrf_1_test_9011.txt",
				"notenw_CflairWglove_lstmCrf_1_dev.txt",
				"notenw_CflairWnon_lstmCrf_1_test_9023.txt",
				"notenw_CflairWnon_lstmCrf_1_dev.txt",
				"notenw_CbertWnon_lstmCrf_1_test_9077.txt",
				"notenw_CbertWnon_lstmCrf_1_dev.txt",
				"notenw_CbertWglove_lstmCrf_1_test_9008.txt",
				"notenw_CbertWglove_lstmCrf_1_dev.txt",
				"notenw_CelmoWnon_lstmCrf_05439651_test_9044.txt",
				"notenw_CelmoWnon_lstmCrf_05439651_dev.txt",
				"notenw_CcnnWglove_lstmCrf_28475920_test_8831.txt",
				"notenw_CcnnWglove_lstmCrf_28475920_dev.txt",
				"notenw_CcnnWglove_cnnCrf_13328976_test_8687.txt",
				"notenw_CcnnWglove_cnnCrf_13328976_dev.txt",
				"notenw_CnonWrand_lstmCrf_62726359_test_8573.txt",
				"notenw_CnonWrand_lstmCrf_62726359_dev.txt",
				"notenw_CcnnWrand_lstmCrf_24879506_test_8603.txt",
				"notenw_CcnnWrand_lstmCrf_24879506_dev.txt",
				"notenw_spanPred_test_9159.txt",
				"notenw_spanPred_dev.txt",

				# base-model
				"spanPred_bert-large-uncased_prunFalse_spLenFalse_spMorphFalse_SpWtFalse_value1_test_9084.txt",
				"spanPred_bert-large-uncased_prunFalse_spLenFalse_spMorphFalse_SpWtFalse_value1_dev.txt",



			]

		elif corpus_type == "notebn":
			fn_prob = "spanPred_bert-large-uncased_prunTrue_spLenFalse_spMorphTrue_SpWtFalse_value1_prob_test_9093.pkl"
			classes = ['PERSON', 'ORG', 'GPE', 'DATE', 'NORP', 'CARDINAL', 'TIME',
					 'LOC', 'FAC', 'PRODUCT', 'WORK_OF_ART', 'MONEY', 'ORDINAL',
					 'QUANTITY', 'EVENT', 'PERCENT', 'LAW', 'LANGUAGE']
			fn_stand_res_test = "notebn_CcnnWglove_lstmCrf_87783107_test_8684.txt"
			fn_stand_res_dev = "notebn_CcnnWglove_lstmCrf_87783107_dev.txt"
			fn_res = [
				"notebn_CflairWglove_lstmCrf_1_test_8943.txt",
				"notebn_CflairWglove_lstmCrf_1_dev.txt",
				"notebn_CflairWnon_lstmCrf_1_test_8903.txt",
				"notebn_CflairWnon_lstmCrf_1_dev.txt",
				"notebn_CbertWnon_lstmCrf_1_test_9053.txt",
				"notebn_CbertWnon_lstmCrf_1_dev.txt",
				"notebn_CbertWglove_lstmCrf_1_test_9070.txt",
				"notebn_CbertWglove_lstmCrf_1_dev.txt",
				"notebn_CcnnWglove_cnnCrf_24059101_test_8618.txt",
				"notebn_CcnnWglove_cnnCrf_24059101_dev.txt",
				"notebn_CelmoWglove_lstmCrf_27696695_test_8933.txt",
				"notebn_CelmoWglove_lstmCrf_27696695_dev.txt",
				"notebn_CcnnWrand_lstmCrf_98713265_test_8387.txt",
				"notebn_CcnnWrand_lstmCrf_98713265_dev.txt",
				"notebn_CelmoWnon_lstmCrf_69707022_test_8921.txt",
				"notebn_CelmoWnon_lstmCrf_69707022_dev.txt",
				"notebn_CcnnWglove_lstmCrf_87783107_test_8684.txt",
				"notebn_CcnnWglove_lstmCrf_87783107_dev.txt",
				"notebn_CnonWrand_lstmCrf_11273623_test_8105.txt",
				"notebn_CnonWrand_lstmCrf_11273623_dev.txt",
				"notebn_spanPred_dev.txt",
				"notebn_spanPred_test_9093.txt",

				"notebn_spanPred1_test_8966.txt",
				"notebn_spanPred1_dev.txt",


			]

		elif corpus_type == "notemz":
			fn_prob = "spanPred_bert-large-uncased_prunTrue_spLenTrue_spMorphFalse_SpWtFalse_value1_prob_test_8703.pkl"
			classes = ['PERSON', 'ORG', 'GPE', 'DATE', 'NORP', 'CARDINAL', 'TIME',
					 'LOC', 'FAC', 'PRODUCT', 'WORK_OF_ART', 'MONEY', 'ORDINAL',
					 'QUANTITY', 'EVENT', 'PERCENT', 'LAW', 'LANGUAGE']
			fn_stand_res_test = "notemz_CnonWrand_lstmCrf_21996043_test_7534.txt"
			fn_stand_res_dev = "notemz_CnonWrand_lstmCrf_21996043_dev.txt"
			fn_res = [
				"notemz_CelmoWnon_lstmCrf_94094287_test_8464.txt",
				"notemz_CelmoWnon_lstmCrf_94094287_dev.txt",
				"notemz_CcnnWglove_cnnCrf_16988381_dev.txt",
				"notemz_CcnnWglove_cnnCrf_16988381_test_8655.txt",
				"notemz_CflairWglove_lstmCrf_1_test_8824.txt",
				"notemz_CflairWglove_lstmCrf_1_dev.txt",
				"notemz_CflairWnon_lstmCrf_1_test_8713.txt",
				"notemz_CflairWnon_lstmCrf_1_dev.txt",
				"notemz_CbertWnon_lstmCrf_1_test_8887.txt",
				"notemz_CbertWnon_lstmCrf_1_dev.txt",
				"notemz_CbertWglove_lstmCrf_1_test_8802.txt",
				"notemz_CbertWglove_lstmCrf_1_dev.txt",
				"notemz_CcnnWrand_lstmCrf_06631052_test_8220.txt",
				"notemz_CcnnWrand_lstmCrf_06631052_dev.txt",
				"notemz_CelmoWglove_lstmCrf_26871338_test_8584.txt",
				"notemz_CelmoWglove_lstmCrf_26871338_dev.txt",
				"notemz_CnonWrand_lstmCrf_21996043_test_7534.txt",
				"notemz_CnonWrand_lstmCrf_21996043_dev.txt",
				"notemz_CcnnWglove_lstmCrf_73202005_test_8661.txt",
				"notemz_CcnnWglove_lstmCrf_73202005_dev.txt",
				"notemz_spanPred_test_8703.txt",
				"notemz_spanPred_dev.txt",

				# base-model
				"notemz_spanPred_prunFalse_spLenTrue_test_8542.txt",
				"notemz_spanPred_prunFalse_spLenTrue_dev.txt"
			]

		elif corpus_type == "notebc":
			fn_prob ="spanPred_bert-large-uncased_prunTrue_spLenFalse_spMorphTrue_SpWtFalse_value1_prob_test_8297.pkl"
			classes = ['PERSON', 'ORG', 'GPE', 'DATE', 'NORP', 'CARDINAL', 'TIME',
					 'LOC', 'FAC', 'PRODUCT', 'WORK_OF_ART', 'MONEY', 'ORDINAL',
					 'QUANTITY', 'EVENT', 'PERCENT', 'LAW', 'LANGUAGE']
			fn_stand_res_test = "notebc_CelmoWnon_lstmCrf_94312093_test_7932.txt"
			fn_stand_res_dev = "notebc_CelmoWnon_lstmCrf_94312093_dev.txt"
			fn_res = [
				"notebc_CcnnWglove_cnnCrf_54677484_test_7463.txt",
				"notebc_CcnnWglove_cnnCrf_54677484_dev.txt",
				"notebc_CcnnWrand_lstmCrf_97843220_test_6981.txt",
				"notebc_CcnnWrand_lstmCrf_97843220_dev.txt",
				"notebc_CnonWrand_lstmCrf_67683313_test_6642.txt",
				"notebc_CnonWrand_lstmCrf_67683313_dev.txt",
				"notebc_CflairWnon_lstmCrf_1_test_7955.txt",
				"notebc_CflairWnon_lstmCrf_1_dev.txt",
				"notebc_CflairWglove_lstmCrf_1_test_7817.txt",
				"notebc_CflairWglove_lstmCrf_1_dev.txt",
				"notebc_CbertWnon_lstmCrf_1_test_8011.txt",
				"notebc_CbertWnon_lstmCrf_1_dev.txt",
				"notebc_CbertWglove_lstmCrf_1_test_8155.txt",
				"notebc_CbertWglove_lstmCrf_1_dev.txt",
				"notebc_CelmoWglove_lstmCrf_58559448_test_7828.txt",
				"notebc_CelmoWglove_lstmCrf_58559448_dev.txt",
				"notebc_CelmoWnon_lstmCrf_94312093_test_7932.txt",
				"notebc_CelmoWnon_lstmCrf_94312093_dev.txt",
				"notebc_CcnnWglove_lstmCrf_65693787_test_7510.txt",
				"notebc_CcnnWglove_lstmCrf_65693787_dev.txt",
				"notebc_spanPred_dev.txt",
				"notebc_spanPred_test_8297.txt",

				#base-model
				"notebc_spanPred_prunTrue_spLenFalse_test_8224.txt",
				"notebc_spanPred_prunTrue_spLenFalse_dev.txt",

			]

		elif corpus_type == "notewb":
			fn_prob = "spanPred_bert-large-uncased_prunTrue_spLenFalse_spMorphTrue_SpWtFalse_value1_prob_test_6858.pkl"
			classes = ['PERSON', 'ORG', 'GPE', 'DATE', 'NORP', 'CARDINAL', 'TIME',
					 'LOC', 'FAC', 'PRODUCT', 'WORK_OF_ART', 'MONEY', 'ORDINAL',
					 'QUANTITY', 'EVENT', 'PERCENT', 'LAW', 'LANGUAGE']
			fn_stand_res_test = "notewb_CcnnWglove_lstmCrf_91552736_test_6261.txt"
			fn_stand_res_dev = "notewb_CcnnWglove_lstmCrf_91552736_dev.txt"
			fn_res = [
				"notewb_CflairWglove_lstmCrf_1_test_6719.txt",
				"notewb_CflairWglove_lstmCrf_1_dev.txt",
				"notewb_CelmoWglove_lstmCrf_00151752_test_6262.txt",
				"notewb_CelmoWglove_lstmCrf_00151752_dev.txt",
				"notewb_CflairWnon_lstmCrf_1_test_6778.txt",
				"notewb_CflairWnon_lstmCrf_1_dev.txt",
				"notewb_CbertWnon_lstmCrf_1_test_6290.txt",
				"notewb_CbertWnon_lstmCrf_1_dev.txt",
				"notewb_CbertWglove_lstmCrf_1_test_6214.txt",
				"notewb_CbertWglove_lstmCrf_1_dev.txt",
				"notewb_CelmoWnon_lstmCrf_09151824_test_6169.txt",
				"notewb_CelmoWnon_lstmCrf_09151824_dev.txt",
				"notewb_CcnnWglove_cnnCrf_19594968_test_4985.txt",
				"notewb_CcnnWglove_cnnCrf_19594968_dev.txt",
				"notewb_CcnnWrand_lstmCrf_68998084_test_5135.txt",
				"notewb_CcnnWrand_lstmCrf_68998084_dev.txt",
				"notewb_CcnnWglove_lstmCrf_91552736_test_6261.txt",
				"notewb_CcnnWglove_lstmCrf_91552736_dev.txt",
				"notewb_CnonWrand_lstmCrf_69309964_test_4891.txt",
				"notewb_CnonWrand_lstmCrf_69309964_dev.txt",
				"notewb_spanPred_test_6858.txt",
				"notewb_spanPred_dev.txt",

				# base-model
				"spanPred_bert-large-uncased_prunTrue_spLenFalse_spMorphFalse_SpWtFalse_value1_test_6792.txt",
				"spanPred_bert-large-uncased_prunTrue_spLenFalse_spMorphFalse_SpWtFalse_value1_dev.txt",


			]

		elif corpus_type == "notetc":
			fn_prob ="spanPred_bert-large-uncased_prunTrue_spLenFalse_spMorphTrue_SpWtFalse_value1_prob_test_6871.pkl"
			classes = ['PERSON', 'ORG', 'GPE', 'DATE', 'NORP', 'CARDINAL', 'TIME',
					 'LOC', 'FAC', 'PRODUCT', 'WORK_OF_ART', 'MONEY', 'ORDINAL',
					 'QUANTITY', 'EVENT', 'PERCENT', 'LAW', 'LANGUAGE']
			fn_stand_res_test = 'notetc_CcnnWglove_lstmCrf_01400817_test_6436.txt'
			fn_stand_res_dev = 'notetc_CcnnWglove_lstmCrf_01400817_dev.txt'
			fn_res = [
				"notetc_CflairWglove_lstmCrf_1_test_6657.txt",
				"notetc_CflairWglove_lstmCrf_1_dev.txt",
				"notetc_CflairWnon_lstmCrf_1_test_6558.txt",
				"notetc_CflairWnon_lstmCrf_1_dev.txt",
				"notetc_CbertWnon_lstmCrf_1_test_7101.txt",
				"notetc_CbertWnon_lstmCrf_1_dev.txt",
				"notetc_CbertWglove_lstmCrf_1_test_7107.txt",
				"notetc_CbertWglove_lstmCrf_1_dev.txt",
				"notetc_CcnnWglove_cnnCrf_65809801_test_5616.txt",
				"notetc_CcnnWglove_cnnCrf_65809801_dev.txt",
				"notetc_CelmoWnon_lstmCrf_36920835_test_6557.txt",
				"notetc_CelmoWnon_lstmCrf_36920835_dev.txt",
				"notetc_CelmoWglove_lstmCrf_46099222_test_6462.txt",
				"notetc_CelmoWglove_lstmCrf_46099222_dev.txt",
				"notetc_CcnnWrand_lstmCrf_26530303_test_5183.txt",
				"notetc_CcnnWrand_lstmCrf_26530303_dev.txt",
				"notetc_CcnnWglove_lstmCrf_01400817_test_6436.txt",
				"notetc_CcnnWglove_lstmCrf_01400817_dev.txt",
				"notetc_CnonWrand_lstmCrf_86707482_dev.txt",
				"notetc_CnonWrand_lstmCrf_86707482_test_4684.txt",
				"notetc_spanPred_test_6871.txt",
				"notetc_spanPred_dev.txt",

				# base-model
				"spanPred_bert-large-uncased_prunFalse_spLenFalse_spMorphFalse_SpWtFalse_value1_test_6667.txt",
				"spanPred_bert-large-uncased_prunFalse_spLenFalse_spMorphFalse_SpWtFalse_value1_dev.txt",


			]

		elif corpus_type == "conll02dutch":
			fn_prob = "spanPred_bert-base-multilingual-uncased_maxSpan4prunTrue_spLenTrue_spMorphFalse_SpWtFalse_value1_prob_test_9104.pkl"
			classes = ["ORG", "PER", "LOC", "MISC"]
			fn_stand_res_test = 'conll02dutch_CcnnWfast_lstmCrf_11426530_test_8223.txt'
			fn_stand_res_dev = 'conll02dutch_CcnnWfast_lstmCrf_11426530_dev.txt'
			fn_res = [
				"conll02dutch_CcnnWrand_lstmCrf_42854051_test_7544.txt",
				"conll02dutch_CcnnWrand_lstmCrf_42854051_dev.txt",
				"conll02dutch_CnonWrand_lstmCrf_18377660_test_6478.txt",
				"conll02dutch_CnonWrand_lstmCrf_18377660_dev.txt",
				"conll02dutch_CbertWnon_lstmCrf_1_test_9119.txt",
				"conll02dutch_CbertWnon_lstmCrf_1_dev.txt",
				"conll02dutch_CbertWfast_lstmCrf_1_test_9087.txt",
				"conll02dutch_CbertWfast_lstmCrf_1_dev.txt",
				"conll02dutch_CflairWnon_lstmCrf_1_test_8711.txt",
				"conll02dutch_CflairWnon_lstmCrf_1_dev.txt",
				"conll02dutch_CflairWglove_lstmCrf_1_test_8776.txt",
				"conll02dutch_CflairWglove_lstmCrf_1_dev.txt",
				"conll02dutch_CcnnWfast_lstmCrf_11426530_test_8223.txt",
				"conll02dutch_CcnnWfast_lstmCrf_11426530_dev.txt",
				"conll02dutch_CcnnWfast_cnnCrf_23521749_test_8070.txt",
				"conll02dutch_CcnnWfast_cnnCrf_23521749_dev.txt",
				"conll02dutch_spanPred_dev.txt",
				"conll02dutch_spanPred_test_9104.txt",

				# base-model
				"conll02dutch_spanPred_prunFalse_spLenFalse_test_8879.txt",
				"conll02dutch_spanPred_prunFalse_spLenFalse_dev.txt",



			]

		elif corpus_type == "conll02spanish":
			fn_prob = "spanPred_bert-base-multilingual-uncased_maxSpan4prunFalse_spLenFalse_spMorphFalse_SpWtFalse_value1_prob_test_8731.pkl"

			classes = ["ORG", "PER", "LOC", "MISC"]
			fn_stand_res_test = 'conll02spanish_CcnnWfast_lstmCrf_09968158_test_8233.txt'
			fn_stand_res_dev = 'conll02spanish_CcnnWfast_lstmCrf_09968158_dev.txt'
			fn_res = [
				"conll02spanish_CelmoWnon_lstmCrf_98074408_test_8311.txt",
				"conll02spanish_CelmoWnon_lstmCrf_98074408_dev.txt",
				"conll02spanish_CelmoWglove_lstmCrf_39383523_test_8370.txt",
				"conll02spanish_CelmoWglove_lstmCrf_39383523_dev.txt",
				"conll02spanish_CcnnWrand_lstmCrf_99264816_test_7944.txt",
				"conll02spanish_CcnnWrand_lstmCrf_99264816_dev.txt",
				"conll02spanish_CnonWrand_lstmCrf_68087567_test_7066.txt",
				"conll02spanish_CnonWrand_lstmCrf_68087567_dev.txt",
				"conll02spanish_CflairWglove_lstmCrf_1_test_8777.txt",
				"conll02spanish_CflairWglove_lstmCrf_1_dev.txt",
				"conll02spanish_CflairWnon_lstmCrf_1_test_8742.txt",
				"conll02spanish_CflairWnon_lstmCrf_1_dev.txt",
				"conll02spanish_CbertWglove_lstmCrf_1_test_8881.txt",
				"conll02spanish_CbertWglove_lstmCrf_1_dev.txt",
				"conll02spanish_CbertWnon_lstmCrf_1_test_8800.txt",
				"conll02spanish_CbertWnon_lstmCrf_1_dev.txt",
				"conll02spanish_CcnnWfast_lstmCrf_09968158_test_8233.txt",
				"conll02spanish_CcnnWfast_lstmCrf_09968158_dev.txt",
				"conll02spanish_CcnnWfast_cnnCrf_97905278_test_8001.txt",
				"conll02spanish_CcnnWfast_cnnCrf_97905278_dev.txt",

				"spanPred_bert-base-multilingual-uncased_maxSpan4prunFalse_spLenFalse_spMorphFalse_SpWtFalse_value1_test_8731.txt",
				"spanPred_bert-base-multilingual-uncased_maxSpan4prunFalse_spLenFalse_spMorphFalse_SpWtFalse_value1_dev.txt",

				#base-model
				"spanPred_bert-base-multilingual-uncased_maxSpan4prunFalse_spLenFalse_spMorphFalse_SpWtFalse_value1_test_8458.txt",
				"spanPred_bert-base-multilingual-uncased_maxSpan4prunFalse_spLenFalse_spMorphFalse_SpWtFalse_value1_dev.txt",

			]
		elif corpus_type == "wnut16":
			fn_prob = "spanPred_bert-large-uncased_prunFalse_spLenFalse_spMorphFalse_SpWtFalse_value1_prob_test_5628.pkl"
			classes = ['loc', "facility", "movie", "company", "product", "person", "other", "tvshow", "musicartist",
					   "sportsteam"]
			fn_stand_res_test = "wnut16_CcnnWglove_lstmCrf_73135194_test_4204.txt"
			fn_stand_res_dev = "wnut16_CcnnWglove_lstmCrf_73135194_dev.txt"
			fn_res = [
				"wnut16_CbertWnon_lstmCrf_1_test_4987.txt",
				"wnut16_CbertWnon_lstmCrf_1_dev.txt",
				"wnut16_CflairWnon_lstmCrf_1_test_5222.txt",
				"wnut16_CflairWnon_lstmCrf_1_dev.txt",
				"wnut16_CflairWglove_lstmCrf_1_test_5207.txt",
				"wnut16_CflairWglove_lstmCrf_1_dev.txt",
				"wnut16_CbertWglove_lstmCrf_1_test_5018.txt",
				"wnut16_CbertWglove_lstmCrf_1_dev.txt",
				"wnut16_CelmoWnon_lstmCrf_39743481_test_4986.txt",
				"wnut16_CelmoWnon_lstmCrf_39743481_dev.txt",
				"wnut16_CelmoWglove_lstmCrf_72013922_test_5022.txt",
				"wnut16_CelmoWglove_lstmCrf_72013922_dev.txt",
				"wnut16_CcnnWglove_lstmCrf_73135194_test_4204.txt",
				"wnut16_CcnnWglove_lstmCrf_73135194_dev.txt",
				"wnut16_CcnnWglove_cnnCrf_20921590_test_3940.txt",
				"wnut16_CcnnWglove_cnnCrf_20921590_dev.txt",
				"wnut16_CcnnWrand_lstmCrf_01735266_test_2068.txt",
				"wnut16_CcnnWrand_lstmCrf_01735266_dev.txt",
				"wnut16_CnonWrand_lstmCrf_22389845_test_1724.txt",
				"wnut16_CnonWrand_lstmCrf_22389845_dev.txt",
				"spanPred_bert-large-uncased_prunFalse_spLenFalse_spMorphFalse_SpWtFalse_value1_test_5628.txt",  # meta
				"spanPred_bert-large-uncased_prunFalse_spLenFalse_spMorphFalse_SpWtFalse_value1_dev.txt",

				# base-model
				"spanPred_bert-large-uncased_prunFalse_spLenFalse_spMorphFalse_SpWtFalse_value1_test_5570.txt",
				"spanPred_bert-large-uncased_prunFalse_spLenFalse_spMorphFalse_SpWtFalse_value1_dev.txt",
			]
		elif corpus_type == "wnut17":
			fn_prob = "spanPred_bert-large-uncased_prunFalse_spLenTrue_spMorphFalse_SpWtFalse_value1_prob_test_5297.pkl"
			classes = ["location", "group", "corporation", "person", "creative-work", "product"]
			fn_stand_res_test = 'wnut17_CcnnWglove_lstmCrf_57496921_test_3641.txt'
			fn_stand_res_dev = 'wnut17_CcnnWglove_lstmCrf_57496921_dev.txt'
			fn_res = [
				"wnut17_CnonWrand_lstmCrf_54345467_test_1839.txt",
				"wnut17_CnonWrand_lstmCrf_54345467_dev.txt",
				"wnut17_CbertWnon_lstmCrf_1_test_4647.txt",
				"wnut17_CbertWnon_lstmCrf_1_dev.txt",
				"wnut17_CcnnWglove_cnnCrf_85832266_test_3372.txt",
				"wnut17_CcnnWglove_cnnCrf_85832266_dev.txt",
				"wnut17_CelmoWglove_lstmCrf_95174997_test_4891.txt",
				"wnut17_CelmoWglove_lstmCrf_95174997_dev.txt",
				"wnut17_CcnnWglove_lstmCrf_57496921_test_3641.txt",
				"wnut17_CcnnWglove_lstmCrf_57496921_dev.txt",
				"wnut17_CcnnWrand_lstmCrf_93380563_test_1877.txt",
				"wnut17_CcnnWrand_lstmCrf_93380563_dev.txt",
				"wnut17_CbertWglove_lstmCrf_1_test_4523.txt",
				"wnut17_CbertWglove_lstmCrf_1_dev.txt",
				"wnut17_CflairWglove_lstmCrf_1_test_4475.txt",
				"wnut17_CflairWglove_lstmCrf_1_dev.txt",
				"wnut17_CflairWnon_lstmCrf_1_test_4357.txt",
				"wnut17_CflairWnon_lstmCrf_1_dev.txt",
				"wnut17_CelmoWnon_lstmCrf_39726004_test_4735.txt",
				"wnut17_CelmoWnon_lstmCrf_39726004_dev.txt",
				"spanPred_bert-large-uncased_prunFalse_spLenTrue_spMorphFalse_SpWtFalse_value1_test_5297.txt",  # ,meta
				"spanPred_bert-large-uncased_prunFalse_spLenTrue_spMorphFalse_SpWtFalse_value1_dev.txt",
				# base-model
				"spanPred_bert-large-uncased_prunFalse_spLenFalse_spMorphFalse_SpWtFalse_value1_test_5205.txt",
				"spanPred_bert-large-uncased_prunFalse_spLenFalse_spMorphFalse_SpWtFalse_value1_dev.txt",
			]

		file_dir = "results2/" +corpus_type


		f1_tests = []
		fn_tests = []
		for fname in fn_res:
			if 'test' in fname:
				f1 = float(fname.split('_')[-1].split('.')[0]) / 10000
				f1_tests.append(f1)
				fn_tests.append(fname)

		# sort the model by descend
		f1_tests, fn_tests = (list(t) for t in zip(*sorted(zip(f1_tests, fn_tests), reverse=True)))
		fn_devs = []
		for f1_test, fn_test in zip(f1_tests, fn_tests):
			prex = '_'.join(fn_test.split('_')[:-2])
			fn_dev = prex+'_dev.txt'
			fn_devs.append(fn_dev)


		# if corpus_type =='conll03': # because the conll03 have 16dataset
		cfn_tests = [fn_tests,
					 fn_tests[:10],
					  fn_tests[:9],
					  fn_tests[:8],
					  fn_tests[:7],
					  fn_tests[:6],
					  fn_tests[:5],
					  fn_tests[:4],
					  fn_tests[:3],
					  fn_tests[:2],
					  fn_tests[2:4],
					  fn_tests[4:6],
					  fn_tests[3:6],
					  fn_tests[1:],
					  fn_tests[2:],
					  fn_tests[3:],
					  fn_tests[4:],
					  fn_tests[5:],
					  fn_tests[6:],
					  fn_tests[7:],
					  fn_tests[8:],
					 fn_tests[9:],
					 fn_tests[10:],
				   ]
		cf1s = [f1_tests,
				f1_tests[:10],
				 f1_tests[:9],
				 f1_tests[:8],
				 f1_tests[:7],
				 f1_tests[:6],
				 f1_tests[:5],
				 f1_tests[:4],
				 f1_tests[:3],
				 f1_tests[:2],
				 f1_tests[2:4],
				 f1_tests[4:6],
				 f1_tests[3:6],
				 f1_tests[1:],
				 f1_tests[2:],
				 f1_tests[3:],
				 f1_tests[4:],
				 f1_tests[5:],
				 f1_tests[6:],
				 f1_tests[7:],
				 f1_tests[8:],
				f1_tests[9:],
				f1_tests[10:],
				]

		# if corpus_type =='conll03': # because the conll03 have 16dataset
		cfn_devs = [fn_devs,
					fn_devs[:10],
					fn_devs[:9],
					fn_devs[:8],
					fn_devs[:7],
					fn_devs[:6],
					fn_devs[:5],
					fn_devs[:4],
					fn_devs[:3],
					fn_devs[:2],
					fn_devs[2:4],
					fn_devs[4:6],
					fn_devs[3:6],
					fn_devs[1:],
					fn_devs[2:],
					fn_devs[3:],
					fn_devs[4:],
					fn_devs[5:],
					fn_devs[6:],
					fn_devs[7:],
					fn_devs[8:],
					fn_devs[9:],
					fn_devs[10:],
					]
		print("cfn_tests:", cfn_tests)
		if len(fn_tests)!=12:
			flag =1
			if corpus_type == "conll02dutch" or corpus_type == "conll02spanish":
				if len(fn_tests)==10:
					flag=0

			if flag==1:
				print('dataname', corpus_type)
				print(len(fn_tests))
				break

		cfn_tests =[fn_tests[:8]]
		cf1s =[f1_tests[:8]]
		cfn_devs =[fn_devs[:8]]

		for cfn_test, cf1 in zip(cfn_tests, cf1s):
			print(cfn_test, cf1)

		result_store_dic = {}
		def result_store(dic, llist, name):
			if name not in dic:
				dic[name] = []
			dic[name].append(llist)
			return dic

		print('args.useSpanLen: ',args.useSpanLen)
		print('args.useMorph: ', args.useMorph)
		print('args.useOverallF1: ', args.useOverallF1)

		comb_ress =[]
		for cfn_test, cf1, cfn_dev in zip(cfn_tests, cf1s, cfn_devs):
			print()
			print('cfn_test', cfn_test)
			combt = CombTraditional(corpus_type, file_dir,useSpanLen=args.useSpanLen, useMorph=args.useMorph,useOverallF1=args.useOverallF1)
			comb_res = combt.get_tradCombinator(corpus_type, file_dir, classes, cfn_dev, fn_stand_res_dev, cfn_test,
							   fn_stand_res_test,cf1)
			comb_ress.append(comb_res)
		comb_results[corpus_type] = comb_ress



		# comb_method =['mlp','svm','xgboost','RandomForest','DecisionTree','KNeighbors','GaussianNB','LR']

	# {'res_normal': res_normal, 'res_prob': res_prob}
	for dataname, comb_ress in comb_results.items():
		print('dataname: ', dataname)
		print('res_normal')
		comb_method = ['svm', 'RandomForest', 'xgboost', 'DecisionTree', 'KNeighbors']
		print(', '.join(comb_method))
		for comb_res in comb_ress:
			res = comb_res['res_normal']
			comb_str = ', '.join(res)
			print(comb_str)

	for dataname, comb_ress in comb_results.items():
		print('dataname: ',dataname)
		print('res_prob')
		comb_method = ['svm', 'RandomForest', 'xgboost', 'DecisionTree', 'KNeighbors']
		print(', '.join(comb_method))
		for comb_res in comb_ress:
			res = comb_res['res_prob']
			comb_str = ', '.join(res)
			print(comb_str)












