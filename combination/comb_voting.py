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
import numpy as np
from collections import Counter
import os
import pickle
import random
from evaluate_metric import get_chunks, get_chunks_onesent,evaluate_chunk_level,evaluate_each_class
import matplotlib.pyplot as plotter

from dataread import DataReader
import matplotlib.mlab as mlab
import matplotlib
if os.environ.get('DISPLAY','') == '':
	print('no display found. Using non-interactive Agg backend')
	matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

class CombByVoting():
	def __init__(self, dataname, file_dir, fmodels, f1s,classes,fn_stand_res,fn_prob):
		self.dataname = dataname
		self.file_dir = file_dir
		self.fmodels = fmodels
		self.f1s = f1s
		self.fn_prob =fn_prob

		self.mres = DataReader(dataname, file_dir,classes,fmodels,fn_stand_res)



		self.wf1 = 1.0
		self.wscore = 0.8

	def get_unique_pchunk_labs(self):
		tchunks_models,\
		tchunks_unique, \
		pchunks_models, \
		tchunks_models_onedim, \
		pchunks_models_onedim, \
		pchunk2label_models, \
		tchunk2label_dic, \
		class2f1_models=self.mres.get_allModels_pred()
		self.tchunks_unique =tchunks_unique
		self.class2f1_models = class2f1_models
		self.tchunk2label_dic = tchunk2label_dic

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
				for i in range(len(self.f1s)):
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
				if tchunk not in keep_pref_upchunks:
					plb_ms = ['O' for i in range(len(self.f1s))]
					pchunk_plb_ms.append(plb_ms)
					keep_pref_upchunks.append(key1)

		return pchunk_plb_ms,keep_pref_upchunks

	def singleModel_error(self):
		# we suppose that the input is one model...
		tchunks_models, \
		tchunks_unique, \
		pchunks_models, \
		tchunks_models_onedim, \
		pchunks_models_onedim, \
		pchunk2label_models, \
		tchunk2label_dic, \
		class2f1_models = self.mres.get_allModels_pred()

		print('num of model inputed: ', len(pchunks_models))
		if len(pchunks_models)>1:
			print('error!!!!')

		# # pchunks = pchunks_models[0]
		# tchunk2label_dic = {}
		# tchunks = tchunks_models[-1] # the span model
		# for tchunk in tchunks:
		# 	tlab, sid, eid, sentid = tchunk
		# 	tkey = (sid, eid, sentid)
		# 	tchunk2label_dic[tkey] =tlab


		print('tchunk2label_dic: ',len(tchunk2label_dic))
		print('len, pchunks', len(pchunks_models[0]))
		elen_num_lists = []
		for n,pchunks in enumerate(pchunks_models):
			errortype_dic = {}
			errorlen_dic ={}
			kchunk_id = []
			for pchunk in pchunks:
				plab, sid, eid, sentid = pchunk
				pkey = (sid, eid, sentid)
				if pkey not in kchunk_id:
					kchunk_id.append(pkey)

					tlab = 'O'
					if pkey in tchunk2label_dic:
						tlab = tchunk2label_dic[pkey]

					if plab != tlab:
						ekey = tlab +'-'+plab
						if ekey not in errortype_dic:
							errortype_dic[ekey] = []
						echunk = (tlab,plab, sid, eid, sentid)
						errortype_dic[ekey].append(echunk)

						# errorlen_dic
						lkey = eid-sid
						if lkey not in errorlen_dic:
							errorlen_dic[lkey] = []
						echunk = (tlab, plab, sid, eid, sentid)
						errorlen_dic[lkey].append(echunk)


			for tchunk in tchunks_unique:
				if tchunk not in pchunks:
					tlab, sid, eid, sentid = tchunk

					tkey = (sid, eid, sentid)
					if tkey not in kchunk_id:
						kchunk_id.append(tkey)

						plab = 'O'
						if plab != tlab:
							ekey = tlab + '-' + plab
							if ekey not in errortype_dic:
								errortype_dic[ekey] = []
							echunk = (tlab, plab, sid, eid, sentid)
							errortype_dic[ekey].append(echunk)

							# errorlen_dic
							lkey = eid - sid
							if lkey not in errorlen_dic:
								errorlen_dic[lkey] = []
							echunk = (tlab, plab, sid, eid, sentid)
							errorlen_dic[lkey].append(echunk)

			etype_num_dic = {}
			for etype, echunk in errortype_dic.items():
				etype_num_dic[etype] = len(echunk)

			etype_num_list = sorted(etype_num_dic.items(), key=lambda x: x[1], reverse=True)

			total_error1 = [num for etype, num in etype_num_list ]
			total_error = np.mean(total_error1)
			print('model %s: '%(n))
			# pieLabels = []
			# populationShare = []
			# for etype, num in etype_num_list:
			# 	# fwrite.write('%s %d\n' % (etype, num))
			# 	print('%s %d' % (etype, num))
			# 	# error_type_count += '%s %d\n' % (etype, num)
			# 	if num<15:
			# 		pieLabels.append('')
			# 	else:
			#
			# 		pieLabels.append(etype)
			# 	# populationShare.append(round())
			#
			# 	populationShare.append(num)
			# # self.draw_etype_pie(populationShare,pieLabels)
			#
			#
			# # draw two-layer pie
			# twolayer_pie_dic = {}
			# hetypes = ['O-ent', 'ent-O','ent-ent']
			# for hetype in hetypes:
			# 	twolayer_pie_dic[hetype] = {}
			# for etype, num in etype_num_list:
			# 	tlab, plab = etype.split('-')
			# 	if tlab=='O': #O->entity
			# 		o_e = 'O-ent'
			# 		twolayer_pie_dic[o_e][etype] = num
			# 	if tlab != 'O' and plab =='O':
			# 		e_o = 'ent-O'
			# 		twolayer_pie_dic[e_o][etype] = num
			# 	if tlab != 'O' and plab !='O':
			# 		e_e = 'ent-ent'
			# 		twolayer_pie_dic[e_e][etype] = num
			# # print('twolayer_pie_dic: ',twolayer_pie_dic)
			# etypeLv1_num_dic = {}
			# total = 0
			# for etype, elem in twolayer_pie_dic.items():
			# 	count = 0
			# 	for e1, num in twolayer_pie_dic[etype].items():
			# 		count+=num
			# 	etypeLv1_num_dic[etype] = count
			# 	total+=count
			# print('total',total)
			# print('O-ent', etypeLv1_num_dic['O-ent'])
			# print('ratio',etypeLv1_num_dic['O-ent']/total)
			# print('ent-O', etypeLv1_num_dic['ent-O'])
			# print('ratio', etypeLv1_num_dic['ent-O'] / total)
			# print('ent-ent', etypeLv1_num_dic['ent-ent'])
			# print('ratio', etypeLv1_num_dic['ent-ent'] / total)
			#
			#
			# for errorlen_dic
			elen_num_dic = {}
			for etype, echunk in errorlen_dic.items():
				elen_num_dic[etype] = len(echunk)

			elen_num_list = sorted(elen_num_dic.items(), key=lambda x: x[1], reverse=True)

			total_error1 = [num for etype, num in elen_num_list]
			# elen_num_lists.append([elen_num_list,total_error1])
			total_e = 0
			for elen, num in elen_num_list:
				total_e+=num
			for elen, num in elen_num_list:
				ratio = num/total_e
				print(elen,num,ratio)





			# group_names = []
			# group_size = []
			# subgroup_names = []
			# subgroup_size = []
			# for hetype, letype in twolayer_pie_dic.items():
			# 	group_size1 = 0
			# 	for etype, num in letype.items():
			# 		group_size1 += num
			# 		if num<20:
			# 			subgroup_names.append('')
			# 		else:
			# 			subgroup_names.append(etype)
			# 		subgroup_size.append(num)
			# 	group_size.append(group_size1)
			# 	group_names.append(hetype)
			#
			# fn_save = ''
			# self.draw_etype_pie_twolayer(group_names, group_size, subgroup_names, subgroup_size, fn_save)
			#



	def best_potential(self):
		pchunk_plb_ms, keep_pref_upchunks = self.get_unique_pchunk_labs()
		print("len(pchunk_plb_ms): ", len(pchunk_plb_ms))
		print("len(keep_upchunks): ", len(keep_pref_upchunks))
		assert len(pchunk_plb_ms) == len(keep_pref_upchunks)


		comb_kchunks = []
		for pchunk_plb_m,pref_upchunks in zip(pchunk_plb_ms, keep_pref_upchunks):
			sid, eid, sentid = pref_upchunks
			key1 = (sid, eid, sentid)

			klb = ''
			# print("pchunk_plb_m: ", pchunk_plb_m)
			if key1 in self.tchunk2label_dic:
				klb = self.tchunk2label_dic[key1]
			elif 'O' in pchunk_plb_m:
				klb = 'O'
			else:
				klb = pchunk_plb_m[0]
			# print(pchunk_plb_m,klb)
			if klb !='O':
				kchunk = (klb, sid, eid, sentid)
				comb_kchunks.append(kchunk)

		f1, p, r, correct_preds, total_preds, total_correct = evaluate_chunk_level(comb_kchunks, self.tchunks_unique)
		print()
		print('best_potential results: ')
		print("f1, p, r, correct_preds, total_preds, total_correct:")
		print(f1, p, r, correct_preds, total_preds, total_correct)

		return [f1, p, r, correct_preds, total_preds, total_correct]


	def voting_majority(self):
		pchunk_plb_ms, keep_pref_upchunks = self.get_unique_pchunk_labs()
		print("len(pchunk_plb_ms): ", len(pchunk_plb_ms))
		print("len(keep_upchunks): ", len(keep_pref_upchunks))
		assert len(pchunk_plb_ms) == len(keep_pref_upchunks)


		comb_kchunks = []
		for pchunk_plb_m,pref_upchunks in zip(pchunk_plb_ms, keep_pref_upchunks):
			lb2num_dic = {}
			for plbm in pchunk_plb_m:
				if plbm not in lb2num_dic:
					lb2num_dic[plbm] = 0.0
				lb2num_dic[plbm] += 1

			klb = sorted(lb2num_dic, key=lambda x: lb2num_dic[x])[-1]
			if klb !='O':
				sid, eid, sentid = pref_upchunks
				kchunk = (klb, sid, eid, sentid)
				comb_kchunks.append(kchunk)
		comb_kchunks = list(set(comb_kchunks))
		f1, p, r, correct_preds, total_preds, total_correct = evaluate_chunk_level(comb_kchunks, self.tchunks_unique)
		print()
		print('majority_voting results: ')
		print("f1, p, r, correct_preds, total_preds, total_correct:")
		print(f1, p, r, correct_preds, total_preds, total_correct)

		kf1 = int(f1 * 10000)
		fn_save_comb_kchunks = 'comb_result/VM_6seq_2span_res' + str(kf1) + '.pkl'

		pickle.dump([comb_kchunks, self.tchunks_unique], open(fn_save_comb_kchunks, "wb"))

		return [f1, p, r, correct_preds, total_preds, total_correct]

	def voting_weightByOverallF1(self):
		pchunk_plb_ms, keep_pref_upchunks = self.get_unique_pchunk_labs()
		assert len(pchunk_plb_ms) == len(keep_pref_upchunks)
		print("len(pchunk_plb_ms): ", len(pchunk_plb_ms))
		print("len(keep_upchunks): ", len(keep_pref_upchunks))

		comb_kchunks = []
		for pchunk_plb_m,pref_upchunks in zip(pchunk_plb_ms, keep_pref_upchunks):
			lb2num_dic = {}
			for plbm,f1 in zip(pchunk_plb_m, self.f1s):
				if plbm not in lb2num_dic:
					lb2num_dic[plbm]=0.0
				lb2num_dic[plbm] += f1

			klb = sorted(lb2num_dic, key=lambda x: lb2num_dic[x])[-1]
			if klb != 'O':
				sid, eid, sentid = pref_upchunks
				kchunk = (klb, sid, eid, sentid)
				comb_kchunks.append(kchunk)

		f1, p, r, correct_preds, total_preds, total_correct = evaluate_chunk_level(comb_kchunks, self.tchunks_unique)
		print()
		print('voting_weightByOverallF1 results: ')
		print("f1, p, r, correct_preds, total_preds, total_correct:")
		print(f1, p, r, correct_preds, total_preds, total_correct)

		kf1 = int(f1 * 10000)
		fn_save_comb_kchunks = 'comb_result/VOF1_6seq_2span_res' + str(kf1) + '.pkl'

		pickle.dump([comb_kchunks, self.tchunks_unique], open(fn_save_comb_kchunks, "wb"))

		return [f1, p, r, correct_preds, total_preds, total_correct]

	def voting_weightByCategotyF1(self):
		pchunk_plb_ms, keep_pref_upchunks = self.get_unique_pchunk_labs()
		assert len(pchunk_plb_ms) == len(keep_pref_upchunks)
		print("len(pchunk_plb_ms): ", len(pchunk_plb_ms))
		print("len(keep_upchunks): ", len(keep_pref_upchunks))

		comb_kchunks = []
		for pchunk_plb_m, pref_upchunks in zip(pchunk_plb_ms, keep_pref_upchunks):
			lb2num_dic = {}
			for plbm, f1, cf1_dic in zip(pchunk_plb_m, self.f1s, self.class2f1_models):
				if plbm not in lb2num_dic:
					lb2num_dic[plbm] = 0.0
				if plbm=='O':
					lb2num_dic[plbm] +=f1
				else:
					lb2num_dic[plbm] += cf1_dic[plbm]

			klb = sorted(lb2num_dic, key=lambda x: lb2num_dic[x])[-1]
			if klb != 'O':
				sid, eid, sentid = pref_upchunks
				kchunk = (klb, sid, eid, sentid)
				comb_kchunks.append(kchunk)

		f1, p, r, correct_preds, total_preds, total_correct = evaluate_chunk_level(comb_kchunks, self.tchunks_unique)
		print()
		print('voting_weightByCategotyF1 results: ')
		print("f1, p, r, correct_preds, total_preds, total_correct:")
		print(f1, p, r, correct_preds, total_preds, total_correct)

		kf1 = int(f1 * 10000)
		fn_save_comb_kchunks = 'comb_result/VCF1_6seq_2span_res' + str(kf1) + '.pkl'

		pickle.dump([comb_kchunks, self.tchunks_unique], open(fn_save_comb_kchunks, "wb"))
		return [f1, p, r, correct_preds, total_preds, total_correct]

	def voting_spanPred_onlyScore(self):
		wf1 = self.wf1
		wscore = self.wscore
		pchunk_plb_ms, keep_pref_upchunks = self.get_unique_pchunk_labs()
		assert len(pchunk_plb_ms) == len(keep_pref_upchunks)
		print("len(pchunk_plb_ms): ", len(pchunk_plb_ms))
		print("len(keep_upchunks): ", len(keep_pref_upchunks))
		print()
		print()
		print('self.fn_prob: ',self.fn_prob)
		pchunk_labPrb_dic = self.mres.read_span_score(keep_pref_upchunks,self.fn_prob)

		comb_kchunks = []
		for pchunk_plb_m, pref_upchunk in zip(pchunk_plb_ms, keep_pref_upchunks):
			lb2num_dic = {}
			for i,(plbm,f1) in enumerate(zip(pchunk_plb_m,self.f1s)):
				if plbm not in lb2num_dic:
					lb2num_dic[plbm] = 0.0
				score = pchunk_labPrb_dic[pref_upchunk][plbm]
				# lb2num_dic[plbm] += score+0.5*f1 # best
				lb2num_dic[plbm] += wscore*score+wf1*f1  # best

			klb = sorted(lb2num_dic, key=lambda x: lb2num_dic[x])[-1]
			if klb != 'O':
				sid, eid, sentid = pref_upchunk
				kchunk = (klb, sid, eid, sentid)
				comb_kchunks.append(kchunk)
		comb_kchunks = list(set(comb_kchunks))




		f1, p, r, correct_preds, total_preds, total_correct = evaluate_chunk_level(comb_kchunks, self.tchunks_unique)
		print()
		print('voting_spanPred_onlyScore results: ')
		print("f1, p, r, correct_preds, total_preds, total_correct:")
		print(f1, p, r, correct_preds, total_preds, total_correct)

		kf1 = int(f1*10000)
		fn_save_comb_kchunks = 'comb_result/SpanNER_6seq_2span_res'+str(kf1)+'.pkl'

		pickle.dump([comb_kchunks,self.tchunks_unique], open(fn_save_comb_kchunks, "wb"))

		return [f1, p, r, correct_preds, total_preds, total_correct]


	def voting_spanPred_spanMLab_NotSeeIn_SeqMLab_conSpanLab(self):
		wf1 = self.wf1
		wscore = self.wscore
		# 当其他模型没有预测的label，span-prediction 预测了，那么就考虑span-prediction；
		# span_modelIdx: [0,1,2], is a list that stores the span-models' index.
		pchunk_plb_ms, keep_pref_upchunks = self.get_unique_pchunk_labs()
		assert len(pchunk_plb_ms) == len(keep_pref_upchunks)
		print("len(pchunk_plb_ms): ", len(pchunk_plb_ms))
		print("len(keep_upchunks): ", len(keep_pref_upchunks))
		pchunk_labPrb_dic = self.mres.read_span_score(keep_pref_upchunks,self.fn_prob)

		comb_kchunks = []
		count_consid_span_model = 0
		for pchunk_plb_m, pref_upchunk in zip(pchunk_plb_ms, keep_pref_upchunks):
			spModel_labs = [pchunk_plb_m[i] for i in self.mres.span_modelIdx]
			seqModel_labs = []
			considModel_idxs = []
			for k in range(len(pchunk_plb_m)):
				if k not in self.mres.span_modelIdx:
					seqModel_labs.append(pchunk_plb_m[k])
					considModel_idxs.append(k) # append the seq model' index.

			# append the span models' index that lab have not seen in the seq-models.
			for sp_mlab, sp_midx in zip(spModel_labs,self.mres.span_modelIdx):
				if sp_mlab not in seqModel_labs:
					considModel_idxs.append(sp_midx)
					count_consid_span_model+=1
					# print("count_consid_span_model: ",count_consid_span_model)

			lb2num_dic = {}
			for midx in considModel_idxs:
				plbm = pchunk_plb_m[midx]
				if plbm not in lb2num_dic:
					lb2num_dic[plbm] = 0.0
				score = pchunk_labPrb_dic[pref_upchunk][plbm]
				f1 = self.f1s[midx]
				lb2num_dic[plbm] += wscore*score+wf1*f1

			klb = sorted(lb2num_dic, key=lambda x: lb2num_dic[x])[-1]
			if klb != 'O':
				sid, eid, sentid = pref_upchunk
				kchunk = (klb, sid, eid, sentid)
				comb_kchunks.append(kchunk)
		comb_kchunks = list(set(comb_kchunks))


		f1, p, r, correct_preds, total_preds, total_correct = evaluate_chunk_level(comb_kchunks, self.tchunks_unique)
		print()
		print("count_consid_span_model: ", count_consid_span_model)
		print('voting_spanPred_spanMLab_NotSeeIn_SeqMLab_conSpanLab results: ')
		print("f1, p, r, correct_preds, total_preds, total_correct:")
		print(f1, p, r, correct_preds, total_preds, total_correct)

		return [f1, p, r, correct_preds, total_preds, total_correct]


	def voting_spanPred_combO_spanMnonO_consSpanLab(self):
		wf1 = self.wf1
		wscore = self.wscore
		# # 其他模型 combination的结果是O; 而span-prediction 是非O, 那么考虑span-predition的；
		# span_modelIdx: [0,1,2], is a list that stores the span-models' index.
		pchunk_plb_ms, keep_pref_upchunks = self.get_unique_pchunk_labs()
		assert len(pchunk_plb_ms) == len(keep_pref_upchunks)
		print("len(pchunk_plb_ms): ", len(pchunk_plb_ms))
		print("len(keep_upchunks): ", len(keep_pref_upchunks))
		pchunk_labPrb_dic = self.mres.read_span_score(keep_pref_upchunks,self.fn_prob)

		comb_kchunks = []
		count_consid_span_model = 0
		for pchunk_plb_m, pref_upchunk in zip(pchunk_plb_ms, keep_pref_upchunks):
			spModel_labs = [pchunk_plb_m[i] for i in self.mres.span_modelIdx]
			seqModel_labs = []
			seqModel_idxs = []
			# spModel_idxs = []
			for k in range(len(pchunk_plb_m)):
				if k not in self.mres.span_modelIdx:
					seqModel_labs.append(pchunk_plb_m[k])
					seqModel_idxs.append(k) # append the seq model' index.

			# first, compute the seq combination result, if the combine res is O,
			# while span-pred models' combinate is non-O, then, use the span-pred lab.

			# combinte the seq model res
			fklb = ''
			seq_klb =''
			sp_klb=''
			all_klb=''
			lb2num_dic = {}
			for midx in seqModel_idxs:
				plbm = pchunk_plb_m[midx]
				if plbm not in lb2num_dic:
					lb2num_dic[plbm] = 0.0
				score = pchunk_labPrb_dic[pref_upchunk][plbm]
				f1 = self.f1s[midx]
				lb2num_dic[plbm] += wscore*score+wf1*f1
			seq_klb = sorted(lb2num_dic, key=lambda x: lb2num_dic[x])[-1]


			if seq_klb != 'O' and len(self.mres.span_modelIdx)!=0: # compute the combination of span model
				sp_lb2num_dic = {}
				for midx in self.mres.span_modelIdx:
					plbm = pchunk_plb_m[midx]
					if plbm not in sp_lb2num_dic:
						sp_lb2num_dic[plbm] = 0.0
					score = pchunk_labPrb_dic[pref_upchunk][plbm]
					f1 = self.f1s[midx]
					sp_lb2num_dic[plbm] += wscore*score+wf1*f1
				# print('sp_lb2num_dic: ',sp_lb2num_dic)
				sp_klb = sorted(sp_lb2num_dic, key=lambda x: sp_lb2num_dic[x])[-1]
				fklb = sp_klb

			else:
				# it means that, when the seq models' combination is non 'O',
				# we will consider all the model's result.
				all_lb2num_dic = {}
				for i, (plbm, f1) in enumerate(zip(pchunk_plb_m, self.f1s)):
					if plbm not in all_lb2num_dic:
						all_lb2num_dic[plbm] = 0.0
					score = pchunk_labPrb_dic[pref_upchunk][plbm]
					# lb2num_dic[plbm] += score+0.5*f1 # best
					all_lb2num_dic[plbm] += wscore*score+wf1*f1  # best

				all_klb = sorted(all_lb2num_dic, key=lambda x: all_lb2num_dic[x])[-1]
				fklb = all_klb
			# print()
			# print(pchunk_plb_m)
			# print("seq_klb: %s,sp_klb: %s,all_klb: %s"%(seq_klb,sp_klb,all_klb))
			if fklb != 'O':
				sid, eid, sentid = pref_upchunk
				kchunk = (fklb, sid, eid, sentid)
				comb_kchunks.append(kchunk)
		comb_kchunks = list(set(comb_kchunks))

		f1, p, r, correct_preds, total_preds, total_correct = evaluate_chunk_level(comb_kchunks, self.tchunks_unique)
		print()
		print("count_consid_span_model: ", count_consid_span_model)
		print('voting_spanPred_combO_spanMnonO_consSpanLab results: ')
		print("f1, p, r, correct_preds, total_preds, total_correct:")
		print(f1, p, r, correct_preds, total_preds, total_correct)

		return [f1, p, r, correct_preds, total_preds, total_correct]


	def draw_etype_pie_twolayer(self,group_names,group_size,subgroup_names,subgroup_size,fn_save):
		# https://www.dreamstime.com/vector-circle-chart-infographic-template-presentations-advertising-layouts-annual-reports-options-steps-parts-vector-circle-image101813915?gclid=EAIaIQobChMI0ZOJvaO-7gIVAQ0qCh1x5AxXEAEYASABEgJOvPD_BwE
		import matplotlib.pyplot as plt

		# # Make data: I have 3 groups and 7 subgroups
		# group_names = ['groupA', 'groupB', 'groupC']
		# group_size = [12, 11, 30]
		# subgroup_names = ['A.1', 'A.2', 'A.3', 'B.1', 'B.2', 'C.1', 'C.2', 'C.3', 'C.4', 'C.5']
		# subgroup_size = [4, 3, 5, 6, 5, 10, 5, 5, 4, 6]

		# Create colors
		a, b, c = [plt.cm.Blues, plt.cm.Reds, plt.cm.Greens]

		# First Ring (outside)
		fig, ax = plt.subplots()
		ax.axis('equal')
		mypie, _ = ax.pie(group_size, radius=1.3, labels=group_names, colors=[a(0.6), b(0.6), c(0.6)])
		plt.setp(mypie, width=0.3, edgecolor='white')

		# Second Ring (Inside)
		mypie2, _ = ax.pie(subgroup_size,
						   radius=1.3 - 0.3,
						   labels=subgroup_names,
						   labeldistance=0.7,
						   # labeldistance=1.1,
						   # autopct='%1.1f%%',
						   startangle=180,
						   pctdistance=0.9,
						   colors=[a(0.5), a(0.4), a(0.3), b(0.5), b(0.4), c(0.6), c(0.5), c(0.4), c(0.3), c(0.2)])
		plt.setp(mypie2, width=0.4, edgecolor='white')
		plt.margins(0, 0)

		# show it
		# plt.show()
		plt.savefig('error_pie/conll03_flair.pdf')

	def draw_etype_pie_bk(self,pieLabels,populationShare):
		#https://pythontic.com/visualization/charts/piechart


		# The slice names of a population distribution pie chart

		# pieLabels = 'Asia', 'Africa', 'Europe', 'North America', 'South America', 'Australia'
		#
		# # Population data
		#
		# populationShare = [59.69, 16, 9.94, 7.79, 5.68, 0.54]

		figureObject, axesObject = plotter.subplots()

		# # Draw the pie chart
		# explodeTuple = (0.1, 0.0, 0.0, 0.0, 0.0, 0.0)
		#
		# # Draw the pie chart
		#
		# axesObject.pie(populationShare, explode=explodeTuple,
		# 			   labels=pieLabels,
		# 			   autopct='%1.2f',
		# 			   startangle=90)

		colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
		axesObject.pie(populationShare,
					   colors=colors,
					   labels=pieLabels,
					   autopct='%1.2f',
					   startangle=90)

		# Aspect ratio - equal means pie is a circle

		axesObject.axis('equal')

		# plotter.show()
		plotter.savefig('error_pie/conll03_flair.pdf')

	def draw_etype_pie(self,xdata,labels):
	# 	colors = ['#ED7D31', '#A5A5A5', '#FFC000', '#5B9BD5', '#71AD47', '#264478', '#9E480D', '#636363', '#997300',
	# '#255E91', '#43682B', '#698ED0', '#F1975A', '#B7B7B7', '#FFCD32', '#8CC168', '#8CC168']
		colors = ['yellowgreen','lightskyblue','thistle','yellow',
				  'pink','orange','powderblue']
		fig2, ax2 = plt.subplots()
		patches_top15, texts_top15, autotexts_top15 = ax2.pie(xdata, labels=labels, colors=colors, labeldistance=1.1, autopct='%1.1f%%', startangle=180, pctdistance=0.9)

		# texts_top15[0].set_fontsize(6)
		# plt.title('Top 15 ' + corpus_type + ' Error Statistic')
		# set foot size.
		proptease_top15 = fm.FontProperties()
		proptease_top15.set_size('large')

		fontsize2 = fm.FontProperties()

		fontsize2.set_size('x-small')

		# ACCEPTS: [size in points | 'xx-small' | 'x-small' | 'small' |
	# 'medium' | 'large' | 'x-large' | 'xx-large' ]
		plt.setp(autotexts_top15, fontproperties=fontsize2)
		plt.setp(texts_top15, fontproperties=proptease_top15)

		plt.show()
		save_path = 'error_pie/conll03_test_top15_%04d.pdf' % random.randint(0, 10000)
		plt.savefig(save_path)
		plt.close()

if __name__ == '__main__':
	# corpus_types = ["conll03"]
	# corpus_types = ["notebn"]
	# corpus_types = ["notebc"]
	corpus_types = ["conll03",  "notenw", "notebn", "notebc", "notemz", "notewb", "notetc"]
	# corpus_types = [ "conll02dutch",
	# 				"conll02spanish"]  # "conll03"
	# corpus_types = ["conll03"] # "conll02dutch", "conll02spanish", "conll02dutch",
	corpus_types = ["notenw", "notebc", "notemz", "notewb", "notetc", "conll02dutch",
					"conll02spanish", "wnut16", "wnut17"]  # "conll03"
	corpus_types = ["conll03"]


	column_true_tag_test, column_pred_tag_test = 1, 2
	classes = []
	fprob = ''
	fn_prob = ''
	fn_stand_res_test = ''
	fn_res = []
	file_dir =''
	for corpus_type in corpus_types:
		# if corpus_type =="conll03":
		# 	# 9318
		# 	# fn_prob = "/home/jlfu/SPred/results/spanPred_dev2train_bert-large-uncased_prunTrue_spLenTrue_spMorphTrue_SpWtFalse_value1_prob_test_9318.pkl"
		# 	# fpath_span = '/home/jlfu/SPred/results/spanPred_dev2train_bert-large-uncased_prunTrue_spLenTrue_spMorphTrue_SpWtFalse_value1_test_9318.txt'  # span-level + pruning
		# 	# #9204
		# 	# fn_prob = "/home/jlfu/mrc/results/conll03_test_prob_9204.pkl"
		# 	# fpath_span = '/home/jlfu/mrc/results/conll03_test_54753855_201230_9205.txt'
		#
		# 	# fn_prob = '/home/jlfu/SPred/combination/results/combinator/conll03_cpu_spanPred_dev2train_bert-large-uncased_prunTrue_spLenTrue_spMorphTrue_SpWtFalse_value1_prob_test_9318.pkl'
		# 	fn_prob = 'combinator_models/conll03/spanPred_dev2train_bert-large-uncased_prunTrue_spLenTrue_spMorphTrue_SpWtFalse_value1_prob_test_9318.pkl'
		#
		# 	# # fn_prob = '/home/jlfu/SPred/combination/results/combinator/conll03_cpu_spanPred_bert-base-uncased_prunTrue_spLenTrue_spMorphTrue_SpWtFalse_value1_prob_test_9201.pkl'
		# 	fn_stand_res = 'conll03_CcnnWglove_lstmCrf_72102467_test_9088.txt'
		# 	print("the file path of probs: ", fn_prob)
		# 	classes = ["ORG", "PER", "LOC", "MISC"]
		# 	fn_res =[
		# 			# "conll03_spanPred_dev2train_bert-large-uncased_prunTrue_spLenTrue_spMorphTrue_SpWtFalse_value1_test_9318.txt",
		# 			# "conll03_luke_test_sentLevel_1_9459.txt",
		# 			# "conll03_xlmr+context-127-best_1_test_9365.txt",
		# 			# "conll03_roberta+context-42-best_1_test_9402.txt",
		#
		# 			"conll03_CflairWglove_lstmCrf_1_test_9302.txt",
		# 			"conll03_CflairWnon_lstmCrf_1_test_9241.txt",
		# 			"conll03_CbertWglove_lstmCrf_1_test_9201.txt",
		# 			"conll03_CbertWnon_lstmCrf_1_test_9246.txt",
		# 			"conll03_CelmoWglove_lstmCrf_95803618_test_9211.txt",
		# 			"conll03_CelmoWnon_lstmCrf_81319158_test_9199.txt",
		# 			"conll03_CcnnWglove_lstmCrf_72102467_test_9088.txt",
		# 			"conll03_CcnnWglove_cnnCrf_45725566_test_8971.txt",
		# 			"conll03_CcnnWrand_lstmCrf_43667285_test_8303.txt",
		# 			"conll03_CnonWrand_lstmCrf_03689925_test_7849.txt",
		#
		# 			# "spanPred_bert-large-uncased_prunTrue_spLenTrue_spMorphFalse_SpWtFalse_value1_test_9245.txt",
		# 			# "spanPred_bert-base-uncased_prunTrue_spLenTrue_spMorphTrue_SpWtFalse_value1_test_9201.txt",
		# 			# "spanPred_bert-base-uncased_prunTrue_spLenFalse_spMorphFalse_SpWstTrue_value0.8_test_9211.txt",
		# 	]

		# elif corpus_type == "notebn":
		# 	# fprob = "/home/jlfu/mrc/results/notebn_test_prob_82075995_210101_9024.pkl"
		# 	# # the path of span-prediction result file
		# 	# fpath_span = '/home/jlfu/mrc/results/notebn_test_82075995_210101_9024.txt'
		# 	fn_prob = 'combinator_models/notenw/spanPred_dev2train_bert-large-uncased_prunTrue_spLenTrue_spMorphTrue_SpWtFalse_value1_prob_test.pkl'
		# 	classes = ['PERSON', 'ORG', 'GPE', 'DATE', 'NORP', 'CARDINAL', 'TIME',
		#              'LOC', 'FAC', 'PRODUCT', 'WORK_OF_ART', 'MONEY', 'ORDINAL',
		#              'QUANTITY','EVENT', 'PERCENT', 'LAW', 'LANGUAGE']
		# 	fn_stand_res ='notenw_CcnnWglove_lstmCrf_28475920_test_8831.txt'
		# 	print("the file path of probs: ", fn_prob)
		# 	fn_res = [
		#
		# 	]
		#
		#
		#
		# elif corpus_type == "notebc":
		# 	# fprob = "/home/jlfu/mrc/results/notebc_test_prob_68734430_210101_8247.pkl"
		# 	# # the path of span-prediction result file
		# 	# fpath_span = '/home/jlfu/mrc/results/notebc_test_68734430_210101_8247.txt'
		# 	classes = ['PERSON', 'ORG', 'GPE', 'DATE', 'NORP', 'CARDINAL', 'TIME',
		# 			 'LOC', 'FAC', 'PRODUCT', 'WORK_OF_ART', 'MONEY', 'ORDINAL',
		# 			 'QUANTITY', 'EVENT', 'PERCENT', 'LAW', 'LANGUAGE']

		if corpus_type == "conll03":
			fn_res = [
				# "conll03_luke_test_sentLevel_1_9459.txt",
				# "conll03_roberta+context-42-best_1_9402.tsv",
				# "conll03_xlmr+context-127-best_1_9365.tsv",
				# "conll03_xlmr+context+dev-43-best_9411.tsv",
				# "conll03_roberta+context+dev-44-best_9411.tsv",

				"conll03_CflairWnon_lstmCrf_1_test_9241.txt",
				# # "conll03_CflairWnon_lstmCrf_1_dev.txt",
				"conll03_CbertWglove_lstmCrf_1_test_9201.txt",
				# # "conll03_CbertWglove_lstmCrf_1_dev.txt",
				"conll03_CbertWnon_lstmCrf_1_test_9246.txt",
				# # "conll03_CbertWnon_lstmCrf_1_dev.txt",
				"conll03_CflairWglove_lstmCrf_1_test_9302.txt",
				# # "conll03_CflairWglove_lstmCrf_1_dev.txt",
				"conll03_CelmoWglove_lstmCrf_95803618_test_9211.txt",
				# # "conll03_CelmoWglove_lstmCrf_95803618_dev.txt",
				"conll03_CcnnWglove_lstmCrf_72102467_test_9088.txt",
				# # "conll03_CcnnWglove_lstmCrf_72102467_dev.txt",
				"conll03_CcnnWglove_cnnCrf_45725566_test_8971.txt",
				# # "conll03_CcnnWglove_cnnCrf_45725566_dev.txt",
				"conll03_CelmoWnon_lstmCrf_81319158_test_9199.txt",
				# # "conll03_CelmoWnon_lstmCrf_81319158_dev.txt",
				"conll03_CnonWrand_lstmCrf_03689925_test_7849.txt",
				# # "conll03_CnonWrand_lstmCrf_03689925_dev.txt",
				"conll03_CcnnWrand_lstmCrf_43667285_test_8303.txt",
				# "conll03_CcnnWrand_lstmCrf_43667285_dev.txt",
				# "conll03_spanPred_test_9245.txt",
				# # "conll03_spanPred_dev.txt",
				# "conll03_spanPred_test_9252.txt",
				# "spanPred_bert-base-uncased_prunTrue_spLenTrue_spMorphTrue_SpWtFalse_value1_test_9201.txt",
				# "spanPred_bert-base-uncased_prunTrue_spLenFalse_spMorphFalse_SpWtFalse_value1_test_9240.txt",
				# "spanPred_bert-large-uncased_prunTrue_spLenTrue_spMorphFalse_SpWtFalse_value1_test_9245.txt",
				# "spanPred_bert-large-uncased_prunTrue_spLenTrue_spMorphFalse_SpWtFalse_value1_test_9246.txt",
				# "spanPred_bert-large-uncased_prunTrue_spLenFalse_spMorphTrue_SpWtFalse_value1_test_9241.txt",
				# "conll03_spanPred_9252_dev.txt",


				"spanPred_prunFalse_spLenFalse_test_9157.txt",
				# "spanPred_new_prunTrue_spLenFalse_test_9189.txt",
				# "spanPred_prunFalse_spLenTrue_test_9222.txt",
				# "spanPred_new_prunTrue_spLenTrue_test_9228.txt",

				# "spanPred_prunTrue_spLenFalse_test_9201.txt",
				# 'spanPred_prunTrue_spLenTrue_test_9195.txt',
				# "spanPred_bert-large-uncased_prunTrue_spLenFalse_spMorphTrue_SpWtFalse_value1_test_9241.txt",



				#dev2train
				"spanPred_1dev2train_bert-large-uncased_prunTrue_spLenFalse_spMorphFalse_SpWtFalse_value1_test_9320.txt",
			]
			fn_stand_res_test = 'conll03_CcnnWglove_lstmCrf_72102467_test_9088.txt'
			fn_stand_res_dev = 'conll03_CcnnWglove_lstmCrf_72102467_dev.txt'
			fn_prob = "spanPred_dev2train_bert-large-uncased_prunTrue_spLenFalse_spMorphFalse_SpWtFalse_value1_prob_test_9320.pkl"
			# fn_prob = "spanPred_bert-large-uncased_prunTrue_spLenTrue_spMorphFalse_SpWtFalse_value1_prob_test_9245.pkl"
			# fn_prob = 'spanPred_dev2train_bert-large-uncased_maxSpan4prunFalse_spLenTrue_spMorphFalse_SpWtFalse_value1_prob_test_9252.pkl'
			classes = ["ORG", "PER", "LOC", "MISC"]
		elif corpus_type == "notenw":
			fn_prob = "spanPred_bert-large-uncased_prunTrue_spLenTrue_spMorphFalse_SpWtFalse_value1_prob_test_9159.pkl"
			classes = ['PERSON', 'ORG', 'GPE', 'DATE', 'NORP', 'CARDINAL', 'TIME',
					   'LOC', 'FAC', 'PRODUCT', 'WORK_OF_ART', 'MONEY', 'ORDINAL',
					   'QUANTITY', 'EVENT', 'PERCENT', 'LAW', 'LANGUAGE']
			fn_stand_res_test = "notenw_CnonWrand_lstmCrf_62726359_test_8573.txt"
			fn_stand_res_dev = "notenw_CnonWrand_lstmCrf_62726359_dev.txt"
			fn_res = [
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
			fn_prob = "spanPred_bert-large-uncased_prunTrue_spLenFalse_spMorphTrue_SpWtFalse_value1_prob_test_8297.pkl"
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

				# base-model
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
			fn_prob = "spanPred_bert-large-uncased_prunTrue_spLenFalse_spMorphTrue_SpWtFalse_value1_prob_test_6871.pkl"
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
				"conll02dutch_spanPred_value1_dev.txt",
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
				# "conll02spanish_CelmoWnon_lstmCrf_98074408_test_8311.txt",
				# "conll02spanish_CelmoWnon_lstmCrf_98074408_dev.txt",
				# "conll02spanish_CelmoWglove_lstmCrf_39383523_test_8370.txt",
				# "conll02spanish_CelmoWglove_lstmCrf_39383523_dev.txt",
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
			classes = ['loc',"facility", "movie", "company", "product", "person", "other", "tvshow", "musicartist", "sportsteam"]
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
				"spanPred_bert-large-uncased_prunFalse_spLenTrue_spMorphFalse_SpWtFalse_value1_test_5297.txt", #,meta
				"spanPred_bert-large-uncased_prunFalse_spLenTrue_spMorphFalse_SpWtFalse_value1_dev.txt",
				# base-model
				"spanPred_bert-large-uncased_prunFalse_spLenFalse_spMorphFalse_SpWtFalse_value1_test_5205.txt",
				"spanPred_bert-large-uncased_prunFalse_spLenFalse_spMorphFalse_SpWtFalse_value1_dev.txt",
			]

		file_dir = "results2/" +corpus_type
		print('file_dir: ',file_dir)
		# fn_prob = os.path.join(file_dir,fn_prob)

		# fn_stand_res =os.path.join(file_dir,fn_stand_res)

		f1s = []
		fnames = []
		for fname in fn_res:
			# fnre = file_dir+'/'+fname
			if '_dev' not in fname:
				f1 = float(fname.split('_')[-1].split('.')[0]) / 10000
				f1s.append(f1)
				fnames.append(fname)


		# sort the model by descend
		f1s, fnames = (list(t) for t in zip(*sorted(zip(f1s, fnames), reverse=True)))
		for f1, fname in zip(f1s, fnames):
			print(f1, fname)

		# if corpus_type =='conll03': # because the conll03 have 16dataset
		cfn_testss = [fnames,
				   fnames[:10],
				   fnames[:9],
				   fnames[:8],
				   fnames[:7],
				   fnames[:6],
				   fnames[:5],
				   fnames[:4],
				   fnames[:3],
				   fnames[:2],
				   fnames[2:4],
				   fnames[4:6],
				   fnames[3:6],
				  fnames[1:],
				  fnames[2:],
				  fnames[3:],
				  fnames[4:],
				  fnames[5:],
				   fnames[6:],
				   fnames[7:],
				   fnames[8:],
				   fnames[9:],
				   fnames[10:]
				   ]
		cf1ss = [f1s,
				 f1s[:10],
				f1s[:9],
				f1s[:8],
				f1s[:7],
				f1s[:6],
				f1s[:5],
				f1s[:4],
				f1s[:3],
				f1s[:2],
				 f1s[2:4],
				 f1s[4:6],
				 f1s[3:6],
				 f1s[1:],
				 f1s[2:],
				 f1s[3:],
				 f1s[4:],
				 f1s[5:],
				 f1s[6:],
				 f1s[7:],
				 f1s[8:],
				 f1s[9:],
				 f1s[10:]
				]

		# cfn_testss = [
		# 			  fnames[:3],
		# 			  fnames[3:7],
		# 			  fnames[7:],
		# 			fnames
		# 			  ]
		# cf1ss = [
		# 		 f1s[:3],
		# 		 f1s[3:7],
		# 		 f1s[7:],
		# 			f1s
		# 		 ]
		if len(fnames) != 12:
			flag = 1
			if corpus_type == "conll02dutch" or corpus_type == "conll02spanish":
				if len(fnames) == 10:
					flag = 0

			if flag == 1:
				print('dataname', corpus_type)
				print(len(fnames))
				break

		# afer use the 2sota model, have 13 models
		cfn_testss = [
			fnames[:8],
			# fnames[4:8],
			# fnames[8:],
			# fnames
		]
		cf1ss = [
			f1s[:8],
			# f1s[4:8],
			# f1s[8:],
			# f1s
		]

		print("cfn_testss:", cfn_testss)
		# cf1ss = [f1s[:3],f1s[:2]]
		# cfn_testss = [fnames[:3],fnames[:2]]

		for fnames, f1s in zip(cfn_testss, cf1ss):
			print(fnames, f1s)

		result_store_dic = {}
		def result_store(dic, llist, name):
			if name not in dic:
				dic[name] = []
			dic[name].append(llist)
			return dic


		print('fn_prob: ', fn_prob)
		fn_prob = os.path.join(file_dir, fn_prob)
		print('fn_prob: ', fn_prob)
		for cfn_tests, cf1s in zip(cfn_testss, cf1ss):
			print()

			print('cfn_tests', cfn_tests)
			mres = DataReader(corpus_type, file_dir, classes, cfn_tests, fn_stand_res_test)

			tchunks_models, tchunks_unique, pchunks_models, tchunks_models_onedim, pchunks_models_onedim, pchunk2label_models, tchunk2label_dic,class2f1_models = mres.get_allModels_pred()
			comvote = CombByVoting(corpus_type, file_dir, cfn_tests, cf1s,classes,fn_stand_res_test,fn_prob)

			# res = comvote.best_potential()
			# result_store_dic = result_store(result_store_dic, res, name='best_potential')

			res = comvote.voting_majority()
			result_store_dic = result_store(result_store_dic, res, name='voting_majority')

			res = comvote.voting_weightByOverallF1()
			result_store_dic = result_store(result_store_dic, res, name='voting_weightByOverallF1')

			res = comvote.voting_weightByCategotyF1()
			result_store_dic = result_store(result_store_dic, res, name='voting_weightByCategotyF1')

			res = comvote.voting_spanPred_onlyScore()
			result_store_dic = result_store(result_store_dic, res, name='voting_spanPred_onlyScore')

			# res = comvote.voting_spanPred_spanMLab_NotSeeIn_SeqMLab_conSpanLab()
			# result_store_dic = result_store(result_store_dic, res, name='voting_spanPred_spanMLab_NotSeeIn_SeqMLab_conSpanLab')
			# # if 15 in idx:
			# # 	continue
			# res = comvote.voting_spanPred_combO_spanMnonO_consSpanLab()
			# result_store_dic = result_store(result_store_dic, res, name='voting_spanPred_combO_spanMnonO_consSpanLab')

		vote_names = []
		resultss= []
		for vote_name,results in result_store_dic.items():

			print("vote_name: ", vote_name)

			kres = []
			for result in results:
				print(result[0])

				kres.append(result[0])
			print()


			# 转置
			vote_names.append(vote_name)
			resultss.append(kres)
		print()
		print('dataset: ', corpus_type)
		print(', '.join(vote_names))
		resultss1 = np.array(resultss).T

		for res1 in resultss1:
			# print(res1)
			res2 = ['%.2f'%(x*100) for x in res1]
			res2 = ', '.join(res2)


			print(res2)




		# # begin{error print}
		# cfn_tests = fn_res
		# cf1s = 0
		# mres = DataReader(corpus_type, file_dir, classes, cfn_tests, fn_stand_res_test)
		# tchunks_models, tchunks_models, pchunks_models, tchunks_models_onedim, pchunks_models_onedim, pchunk2label_models, tchunk2label_dic,class2f1_models = mres.get_allModels_pred()
		# comvote = CombByVoting(corpus_type, file_dir, cfn_tests, cf1s,classes,fn_stand_res_test,fn_prob)
		# comvote.singleModel_error()
		# # end{error print}









