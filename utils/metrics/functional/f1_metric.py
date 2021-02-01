# encoding: utf-8


import torch
from utils.bmes_decode import bmes_decode

import json
import random
from tokenizers import BertWordPieceTokenizer

def span_f1(n_samps,n_classes, all_span_rep,pos_span_mask_ltoken,real_span_mask_ltoken,words,all_span_word,flat=True):
    '''
    :param all_span_rep: the score of model predict. SHAPE: (batch_size,n_span)
    :param all_span_idxs_ltoken: the label of the spans.  SHAPE: (batch_size,n_span)
    :param real_span_mask_ltoken: 1 for real span, and 0 for padding span.
    :return:
    '''
   
    n_span = all_span_rep.size(-1)
    all_span_rep = all_span_rep.view(n_samps,n_classes,n_span).permute(0,2,1)
    all_span_idxs_ltoken = pos_span_mask_ltoken.view(n_samps, n_classes, n_span)
    all_span_idxs_ltoken = all_span_idxs_ltoken.permute(0, 2, 1)

    max_scores = torch.max(all_span_rep, dim=-1)[0]
    max_scores = max_scores.unsqueeze(-1).expand(n_samps,n_span,n_classes)
    label_pred = torch.eq(all_span_rep, max_scores) # (n_samps, n_span, n_classes), 0 for non-max-score, 1 for max-score.

    nonO_max_scores = max_scores[:,:,1:]
    nonO_label_pred = label_pred[:,:,1:].bool()
    nonO_right_label = all_span_idxs_ltoken[:,:,1:].bool()

    correct_pred = (nonO_label_pred & nonO_right_label).long().sum()
    total_pred = nonO_label_pred.long().sum()
    total_golden = nonO_right_label.long().sum()

    return torch.stack([correct_pred, total_pred, total_golden])




def span_f1_spanPred(n_samps,n_classes, predicts,span_label_ltoken,real_span_mask_ltoken,words,all_span_word,flat=True):
    '''
    :param all_span_rep: the score of model predict. SHAPE: (batch_size,n_span)
    :param all_span_idxs_ltoken: the label of the spans.  SHAPE: (batch_size,n_span)
    :param real_span_mask_ltoken: 1 for real span, and 0 for padding span.

    span_label_ltoken: （bs, n_span）
    :return:
    '''
    pred_label_idx = torch.max(predicts, dim=-1)[1] # (bs, n_span)
    pred_label_mask = (pred_label_idx!=0)  # (bs, n_span)

    all_correct = pred_label_idx == span_label_ltoken
    all_correct = all_correct*pred_label_mask*real_span_mask_ltoken.bool()
    correct_pred = torch.sum(all_correct)
    total_pred = torch.sum(pred_label_idx!=0 )
    total_golden = torch.sum(span_label_ltoken!=0)

    return torch.stack([correct_pred, total_pred, total_golden])



def span_f1_spanPred_pruning(all_span_idxs,predicts,span_label_ltoken,real_span_mask_ltoken,words,all_span_word,flat=True):
    '''
    :param all_span_rep: the score of model predict. SHAPE: (batch_size,n_span)
    :param all_span_idxs_ltoken: the label of the spans.  SHAPE: (batch_size,n_span)
    :param real_span_mask_ltoken: 1 for real span, and 0 for padding span.

    span_label_ltoken: （bs, n_span）
    :return:
    '''
    pred_label_idx = torch.max(predicts, dim=-1)[1] # (bs, n_span)
    span_probs = predicts.tolist()
    nonO_idxs2labs, nonO_kidxs_all, pred_label_idx_new = get_pruning_predIdxs(pred_label_idx, all_span_idxs, span_probs)
    pred_label_idx = pred_label_idx_new.cuda()

    pred_label_mask = (pred_label_idx!=0)  # (bs, n_span)

    all_correct = pred_label_idx == span_label_ltoken
    all_correct = all_correct*pred_label_mask*real_span_mask_ltoken.bool()
    correct_pred = torch.sum(all_correct)
    total_pred = torch.sum(pred_label_idx!=0 )
    total_golden = torch.sum(span_label_ltoken!=0)

    return torch.stack([correct_pred, total_pred, total_golden]),pred_label_idx

def get_predict_spanPred(args,all_span_word, words,predicts,span_label_ltoken,all_span_idxs):
    '''
    :param all_span_rep: the score of model predict. SHAPE: (batch_size,n_span)
    :param all_span_idxs_ltoken: the label of the spans.  SHAPE: (batch_size,n_span)
    :param real_span_mask_ltoken: 1 for real span, and 0 for padding span.

    span_label_ltoken: （bs, n_span）
    :return:
    '''
    pred_label_idx = torch.max(predicts, dim=-1)[1] # (bs, n_span)
    # for context
    idx2label = {}
    label2idx_list = args.label2idx_list
    for labidx in label2idx_list:
        lab, idx = labidx
        idx2label[int(idx)] = lab

    batch_preds = []
    for span_idxs,word,ws,lps,lts in zip(all_span_idxs,words,all_span_word,pred_label_idx,span_label_ltoken):
        text = ' '.join(word) +"\t"
        for sid,w,lp,lt in zip(span_idxs,ws,lps,lts):
            if lp !=0 or lt!=0:
                plabel = idx2label[int(lp.item())]
                tlabel = idx2label[int(lt.item())]
                sidx, eidx = sid
                ctext = ' '.join(w)+ ':: '+str(int(sidx))+','+str(int(eidx+1))  +':: '+tlabel +':: '+plabel +'\t'
                text +=ctext
        batch_preds.append(text)
    return batch_preds


def get_predict_spanPred_prune(args,all_span_word, words,predicts_new,span_label_ltoken,all_span_idxs):
    '''
    :param all_span_rep: the score of model predict. SHAPE: (batch_size,n_span)
    :param all_span_idxs_ltoken: the label of the spans.  SHAPE: (batch_size,n_span)
    :param real_span_mask_ltoken: 1 for real span, and 0 for padding span.

    span_label_ltoken: （bs, n_span）
    :return:predicts_new (bs, n_span)
    '''
  
    idx2label = {}
    label2idx_list = args.label2idx_list
    for labidx in label2idx_list:
        lab, idx = labidx
        idx2label[int(idx)] = lab

    batch_preds = []
    for span_idxs,word,ws,lps,lts in zip(all_span_idxs,words,all_span_word,predicts_new,span_label_ltoken):
        text = ' '.join(word) +"\t"
        for sid,w,lp,lt in zip(span_idxs,ws,lps,lts):
            if lp !=0 or lt!=0:
                plabel = idx2label[int(lp.item())]
                tlabel = idx2label[int(lt.item())]
                sidx, eidx = sid
                ctext = ' '.join(w)+ ':: '+str(int(sidx))+','+str(int(eidx+1))  +':: '+tlabel +':: '+plabel +'\t'
                text +=ctext
        batch_preds.append(text)
    return batch_preds

def write_predict_spanPred(dataname,fwrite,all_span_word, words,all_span_idxs_ltoken,predicts,span_label_ltoken,all_span_idxs):
    '''
    :param all_span_rep: the score of model predict. SHAPE: (batch_size,n_span)
    :param all_span_idxs_ltoken: the label of the spans.  SHAPE: (batch_size,n_span)
    :param real_span_mask_ltoken: 1 for real span, and 0 for padding span.

    span_label_ltoken: （bs, n_span）
    :return:
    '''
   
    pred_label_idx = torch.max(predicts, dim=-1)[1] # (bs, n_span)
    label2idx = {}
    if dataname == 'conll03':
        label2idx = {"O": 0, "ORG": 1, "PER": 2, "LOC": 3, "MISC": 4}
    elif 'note' in dataname:
        label2idx = {'O': 0, 'PERSON': 1, 'ORG': 2, 'GPE': 3, 'DATE': 4, 'NORP': 5, 'CARDINAL': 6, 'TIME': 7,
                     'LOC': 8, 'FAC': 9, 'PRODUCT': 10, 'WORK_OF_ART': 11, 'MONEY': 12, 'ORDINAL': 13,
                     'QUANTITY': 14,
                     'EVENT': 15, 'PERCENT': 16, 'LAW': 17, 'LANGUAGE': 18}
    idx2label ={}
    for lab,idx in label2idx.items():
        idx2label[idx] = lab

    batch_preds = []
    for span_idxs,word,ws,lps,lts in zip(all_span_idxs,words,all_span_word,pred_label_idx,span_label_ltoken):
        text = ' '.join(word) +"\t"
        for sid,w,lp,lt in zip(span_idxs,ws,lps,lts):
            if lp !=0 or lt!=0:
                plabel = idx2label[int(lp.item())]
                tlabel = idx2label[int(lt.item())]
                sidx, eidx = sid
                ctext = ' '.join(w)+ ':: '+str(int(sidx))+','+str(int(eidx+1))  +':: '+tlabel +':: '+plabel +'\t'
                text +=ctext
        batch_preds.append(text)
        fwrite.write(text+'\n')

def has_overlapping(idx1, idx2):
    overlapping = True
    if (idx1[0] > idx2[1] or idx2[0] > idx1[1]):
        overlapping = False
    return overlapping

def clean_overlapping_span(idxs_list,nonO_idxs2prob):
    kidxs = []
    didxs = []
    for i in range(len(idxs_list)-1):
        idx1 = idxs_list[i]

        kidx = idx1
        kidx1 = True
        for j in range(i+1,len(idxs_list)):
            idx2 = idxs_list[j]
            isoverlapp = has_overlapping(idx1, idx2)




            if isoverlapp:
                prob1 = nonO_idxs2prob[idx1]
                prob2 = nonO_idxs2prob[idx2]

                if prob1 < prob2:
                    kidx1 = False
                    didxs.append(kidx1)
                elif prob1 == prob2:
                    len1= idx1[1] - idx1[0]+1
                    len2 = idx1[1] - idx1[0] + 1
                    # print("len1, len2: ", len1, len2)
                    if len1<len2:
                        kidx1 = False
                        didxs.append(kidx1)

        if kidx1:
            flag=True
            for idx in kidxs:
                isoverlap= has_overlapping(idx1,idx)
                if isoverlap:
                    flag=False
                    prob1 = nonO_idxs2prob[idx1]
                    prob2 = nonO_idxs2prob[idx]
                    if prob1>prob2: # del the keept idex
                        kidxs.remove(idx)
                        kidxs.append(idx1)
                    break
            if flag==True:
                kidxs.append(idx1)







    if len(didxs)==0:
        kidxs.append(idxs_list[-1])
    else:
        if idxs_list[-1] not in didxs:
            kidxs.append(idxs_list[-1])

    return kidxs

def get_pruning_predIdxs(pred_label_idx, all_span_idxs,span_probs):
    nonO_kidxs_all = []
    nonO_idxs2labs = []
    # begin{Constraint the span that was predicted can not be overlapping.}
    for i, (bs, idxs) in enumerate(zip(pred_label_idx, all_span_idxs)):
        pred_label_idx_new1 = []

        # collect the span indexs that non-O
        nonO_idxs2lab = {}
        nonO_idxs2prob = {}
        nonO_idxs = []
        for j, (plb, idx) in enumerate(zip(bs, idxs)):
            plb = int(plb.item())
            nplb = 0
            if plb != 0:  # only consider the non-O label span...
                nonO_idxs2lab[idx] = plb
                nonO_idxs2prob[idx] = span_probs[i][j][plb]
                nonO_idxs.append(idx)

        nonO_idxs2labs.append(nonO_idxs2lab)
        if len(nonO_idxs) != 0:
            nonO_kidxs = clean_overlapping_span(nonO_idxs, nonO_idxs2prob)
        else:
            nonO_kidxs = []
        nonO_kidxs_all.append(nonO_kidxs)

    pred_label_idx_new = []
    n_span = pred_label_idx.size(1)
    for i, (bs, idxs) in enumerate(zip(pred_label_idx, all_span_idxs)):
        pred_label_idx_new1 = []
        for j, (plb, idx) in enumerate(zip(bs, idxs)):
            nlb_id = 0
            if idx in nonO_kidxs_all[i]:
                nlb_id = plb
            pred_label_idx_new1.append(nlb_id)
        while len(pred_label_idx_new1) <n_span:
            pred_label_idx_new1.append(0)

        pred_label_idx_new.append(pred_label_idx_new1)
    # print('pred_label_idx_new: ',pred_label_idx_new)
    # print('pred_label_idx.shape: ', pred_label_idx.shape)
    pred_label_idx_new = torch.LongTensor(pred_label_idx_new)
    return nonO_idxs2labs,nonO_kidxs_all,pred_label_idx_new


def write_predict_spanPred_pruning(dataname,fwrite,all_span_word, words,all_span_idxs_ltoken,predicts,span_label_ltoken,all_span_idxs):
    '''
    constraint the span that was predicted can not be overlapping.
    :param all_span_rep: the score of model predict. SHAPE: (batch_size,n_span)
    :param all_span_idxs_ltoken: the label of the spans.  SHAPE: (batch_size,n_span)
    :param real_span_mask_ltoken: 1 for real span, and 0 for padding span.

    span_label_ltoken: （bs, n_span）
    :return:
    '''

    pred_label_idx = torch.max(predicts, dim=-1)[1] # (bs, n_span)
    span_probs = predicts.tolist()
    nonO_idxs2labs, nonO_kidxs_all, pred_label_idx_new = get_pruning_predIdxs(pred_label_idx, all_span_idxs,span_probs)


    count_delete = 0
    for i in range(len(nonO_idxs2labs)):
        if len(nonO_idxs2labs[i]) != len(nonO_kidxs_all[i]):
            count_delete+=1
    # end{Constraint the span that was predicted can not be overlapping.}




    label2idx = {}
    if dataname == 'conll03':
        label2idx = {"O": 0, "ORG": 1, "PER": 2, "LOC": 3, "MISC": 4}
    elif 'note' in dataname:
        label2idx = {'O': 0, 'PERSON': 1, 'ORG': 2, 'GPE': 3, 'DATE': 4, 'NORP': 5, 'CARDINAL': 6, 'TIME': 7,
                     'LOC': 8, 'FAC': 9, 'PRODUCT': 10, 'WORK_OF_ART': 11, 'MONEY': 12, 'ORDINAL': 13,
                     'QUANTITY': 14,
                     'EVENT': 15, 'PERCENT': 16, 'LAW': 17, 'LANGUAGE': 18}

    idx2label ={}
    for lab,idx in label2idx.items():
        idx2label[idx] = lab




    for span_idxs,word,ws,lps,lts in zip(all_span_idxs,words,all_span_word,pred_label_idx_new,span_label_ltoken):
        text = ' '.join(word) +"\t"
        for sid,w,lp,lt in zip(span_idxs,ws,lps,lts):
            if lp !=0 or lt!=0:
                plabel = idx2label[int(lp.item())]
                tlabel = idx2label[int(lt.item())]
                sidx, eidx = sid
                ctext = ' '.join(w)+ ':: '+str(int(sidx))+','+str(int(eidx+1))  +':: '+tlabel +':: '+plabel +'\t'
                text +=ctext
        fwrite.write(text+'\n')

    # write the result file as a bio format..







def save_predict_results(n_samps,n_classes,all_span_idxs_ltoken, all_span_word, label_pred,words):
    '''
    :param all_span_word: n_samps, num of spans for a samples
    :param label_pred: SHAPE: (n_samps, n_span, n_classes), 0 for non-max-score, 1 for max-score.
    :param all_span_idxs_ltoken: SHAPE: (n_samps, n_span, 2),
    :return:
    '''
    context = " ".join(words)
    path_en_conll03 = "../data_preprocess/queries/en_conll03.json"
    query_en_conll03 = load_query_map(path_en_conll03)
    labels = query_en_conll03['labels']
    idx2label = {i:label for i,label in enumerate(labels)}
    print(idx2label)

    span_label_idx = torch.max(label_pred,dim=-1) # (n_samps, n_span)

    for i in range(n_samps):
        idx = i*n_classes
        all_span_word1 = all_span_word[idx]
        all_span_label = []
        for j,span_label in enumerate(span_label_idx[i]):
            all_span_label.append(idx2label[span_label])
            sidx, eidx = all_span_idxs_ltoken[i][j]
    print('context: ', context)















def load_query_map(query_map_path):
    with open(query_map_path, "r") as f:
        query_map = json.load(f)

    return query_map




def query_span_f1(start_preds, end_preds, match_logits, start_label_mask, end_label_mask, match_labels, flat=False):
    """
    Compute span f1 according to query-based model output
    Args:
        start_preds: [bsz, seq_len]
        end_preds: [bsz, seq_len]
        match_logits: [bsz, seq_len, seq_len]
        start_label_mask: [bsz, seq_len]
        end_label_mask: [bsz, seq_len]
        match_labels: [bsz, seq_len, seq_len]
        flat: if True, decode as flat-ner
    Returns:
        span-f1 counts, tensor of shape [3]: tp, fp, fn
    """
    start_label_mask = start_label_mask.bool()
    end_label_mask = end_label_mask.bool()
    match_labels = match_labels.bool()
    bsz, seq_len = start_label_mask.size()
    # [bsz, seq_len, seq_len]
    match_preds = match_logits > 0
    # [bsz, seq_len]
    start_preds = start_preds.bool()
    # [bsz, seq_len]
    end_preds = end_preds.bool()

    match_preds = (match_preds
                   & start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                   & end_preds.unsqueeze(1).expand(-1, seq_len, -1))
    match_label_mask = (start_label_mask.unsqueeze(-1).expand(-1, -1, seq_len)
                        & end_label_mask.unsqueeze(1).expand(-1, seq_len, -1))
    match_label_mask = torch.triu(match_label_mask, 0)  # start should be less or equal to end
    match_preds = match_label_mask & match_preds

    tp = (match_labels & match_preds).long().sum()
    fp = (~match_labels & match_preds).long().sum()
    fn = (match_labels & ~match_preds).long().sum()
    return torch.stack([tp, fp, fn])


def extract_flat_spans(start_pred, end_pred, match_pred, label_mask):
    """
    Extract flat-ner spans from start/end/match logits
    Args:
        start_pred: [seq_len], 1/True for start, 0/False for non-start
        end_pred: [seq_len, 2], 1/True for end, 0/False for non-end
        match_pred: [seq_len, seq_len], 1/True for match, 0/False for non-match
        label_mask: [seq_len], 1 for valid boundary.
    Returns:
        tags: list of tuple (start, end)
    Examples:
        >>> start_pred = [0, 1]
        >>> end_pred = [0, 1]
        >>> match_pred = [[0, 0], [0, 1]]
        >>> label_mask = [1, 1]
        >>> extract_flat_spans(start_pred, end_pred, match_pred, label_mask)
        [(1, 2)]
    """
    pseudo_tag = "TAG"
    pseudo_input = "a"

    bmes_labels = ["O"] * len(start_pred)
    start_positions = [idx for idx, tmp in enumerate(start_pred) if tmp and label_mask[idx]]
    end_positions = [idx for idx, tmp in enumerate(end_pred) if tmp and label_mask[idx]]

    for start_item in start_positions:
        bmes_labels[start_item] = f"B-{pseudo_tag}"
    for end_item in end_positions:
        bmes_labels[end_item] = f"E-{pseudo_tag}"

    for tmp_start in start_positions:
        tmp_end = [tmp for tmp in end_positions if tmp >= tmp_start]
        if len(tmp_end) == 0:
            continue
        else:
            tmp_end = min(tmp_end)
        if match_pred[tmp_start][tmp_end]:
            if tmp_start != tmp_end:
                for i in range(tmp_start+1, tmp_end):
                    bmes_labels[i] = f"M-{pseudo_tag}"
            else:
                bmes_labels[tmp_end] = f"S-{pseudo_tag}"

    tags = bmes_decode([(pseudo_input, label) for label in bmes_labels])

    return [(tag.begin, tag.end) for tag in tags]


def remove_overlap(spans):
    """
    remove overlapped spans greedily for flat-ner
    Args:
        spans: list of tuple (start, end), which means [start, end] is a ner-span
    Returns:
        spans without overlap
    """
    output = []
    occupied = set()
    for start, end in spans:
        if any(x for x in range(start, end+1)) in occupied:
            continue
        output.append((start, end))
        for x in range(start, end + 1):
            occupied.add(x)
    return output
