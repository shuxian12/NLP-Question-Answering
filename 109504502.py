# %%
# from .autonotebook import tqdm as notebook_tqdm
import torch
import torch.nn as nn
from io import open
import glob
import os
import re
import numpy as np2
import nltk
from nltk import word_tokenize
# nltk.download('punkt')
import time

# %%
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# %% [markdown]
# ## bert model

# %%
from transformers import BertTokenizerFast, BertModel, DataCollatorWithPadding
MODEL_NAME = 'deepset/bert-base-uncased-squad2'
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
# model

# %%
class QAModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        self.qa_logits = nn.Linear(hidden_size, 2)  # 2 for start and end logits
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, start_positions=None, end_positions=None):
        # attention_mask: mask for padding tokens
        # token_type_ids: mask for context and question
        bert_output = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        
        # output = (last_hidden_state, pooled_output([CLS]), (hidden_states), (attentions))
        logits = self.qa_logits(bert_output[0])     # (batch_size, seq_len, 2)
        
        start_logits, end_logits = logits.split(1, dim=-1)  # (batch_size, seq_len, 1) , dim = -1 means last dimension
        start_logits = start_logits.squeeze(-1)      # (batch_size, seq_len) , axis = -1 removes the last dimension
        end_logits = end_logits.squeeze(-1)
        
        output = (start_logits, end_logits)
        total_loss = None
        if start_positions is not None and end_positions is not None: # do training
            ignored_index = start_logits.size(1)
            # clamp: limit the value between 0 and ignored_index
            start_positions = start_positions.clamp(0, ignored_index)   
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fn = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fn(start_logits, start_positions)
            end_loss = loss_fn(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            output = (total_loss,) + output
        
        return output     # (total_loss), start_logits, end_logits
# %% [markdown]
# ## data

# %%
from dataclasses import dataclass
@dataclass
class QA:
    question: str
    ans: str
    snippets: list

# %%
train_path = 'NLP-2-Dataset/train.txt'
val_path = 'NLP-2-Dataset/val.txt'
test_path = 'NLP-2-Dataset/test.txt'
def read_data(path):
    examples = []
    count = 0
    with open(path, 'r', encoding= 'utf8') as file:
        for line in file.readlines():
            snippets, question, answer = line.split('|||')
            question = question[1:-1]
            answer = answer.split('\n')[0][1:]
            examples.append(QA(question= question,
                                    ans= answer,
                                    snippets=[token[4:-5] for token in re.findall(r"<s>[^<>]*<\/s>", snippets)]))
            count += 1
            
    print(count)
    return examples
# %%
import pickle

train_data_path = 'NLP-2-Dataset/train_data'
val_data_path = 'NLP-2-Dataset/val_data'
test_data_path = 'NLP-2-Dataset/test_data'

def pet_save(path, data):
    with open(path + '.pickle', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def digimon_load(path):
    with open(path + '.pickle', 'rb') as f:
        return pickle.load(f)

# pet_save(train_data_path, test)
# pet_save(val_data_path, test)
# pet_save(test_data_path, test)

# %%
train_data = digimon_load(train_data_path)
val_data = digimon_load(val_data_path)
test_data = digimon_load(test_data_path)

# %% [markdown]
# ## BM25

# %%
from gensim import corpora
from gensim.summarization import bm25

# %% [markdown]
# ### process

# %%
def BM25(datas: list, is_train: bool = True):
	no_catch = 0
	catch = 0
	article_list = []
	ans_snippets = []
	for (i, data) in enumerate(datas):
		querys = data.question
		snippets = data.snippets
		ans = data.ans
		article_list.clear()
		for a in snippets:
			stemmed_tokens = word_tokenize(a)   #[p_stemmer.stem(i) for i in a_split]  
			article_list.append(stemmed_tokens)
		# bm25模型
		bm25Model = bm25.BM25(article_list)
		average_idf = sum(map(lambda k: float(bm25Model.idf[k]), bm25Model.idf.keys())) / len(bm25Model.idf.keys())
		scores = bm25Model.get_scores(querys, average_idf)
		a = torch.tensor(scores)
		
		has_ans_idx, has_no_idx = [], []
		flag = False
		count = 0
		# print(idx)
		if is_train:
			v, idx = a.topk(k=15, largest=True)
			for q in idx:
				line = snippets[q]
				if len(has_ans_idx) > 4 and len(has_no_idx) > 2:
					break
				if ans in line and len(has_ans_idx) < 5:
					has_ans_idx.append([q.item(), line.find(ans), line.find(ans)+len(ans)-1])
				elif len(has_no_idx) < 3:
					has_no_idx.append([q.item(), -1, -1])
			if len(has_ans_idx) >= 2 and len(has_no_idx) >= 2:
				ans_snippets.append(has_ans_idx[:2] + has_no_idx[:2])
			elif len(has_ans_idx) > len(has_no_idx):
				ans_snippets.append(has_ans_idx[:4-len(has_no_idx)] + has_no_idx)
			elif len(has_ans_idx) < len(has_no_idx):
				ans_snippets.append(has_ans_idx + has_no_idx[:4-len(has_ans_idx)])
		else:
			v, idx = a.topk(k=16, largest=True)
			has_ans_idx = idx
			ans_snippets.append(has_ans_idx)
			
            
	return ans_snippets

# %%
# pet_save('NLP-2-Dataset/test_BM25_all', BM25(test_data, False))
# train_bm25 = digimon_load('NLP-2-Dataset/test_BM25')

# %%
# train_data  # question, ans, snippets
# train_bm25  # [snippet_idx, ans_start_idx, ans_end_idx]

def create_bm25_data(path, unprocess_data, is_train=True):
    bm25_datas = digimon_load(path)
    training_question = []
    training_start_idx = []
    trainging_end_idx = []
    training_snippet = []
    traing_ans = []
    for (i, data) in enumerate(unprocess_data):
        bm25_data = bm25_datas[i]
        query = data.question
        snippets = data.snippets
        ans = data.ans
        if is_train:
            if bm25_data[0][1] != -1:
                training_question.extend([query, query])
                training_snippet.extend([snippets[bm25_data[0][0]], snippets[bm25_data[2][0]]])
                training_start_idx.extend([bm25_data[0][1], bm25_data[2][1]])
                trainging_end_idx.extend([bm25_data[0][2], bm25_data[2][2]])
                traing_ans.extend([ans, ''])
            else:
                training_question.extend([query])
                training_snippet.extend([snippets[bm25_data[0][0]]])
                training_start_idx.extend([bm25_data[0][1]])
                trainging_end_idx.extend([bm25_data[0][2]])
                traing_ans.extend([''])
        else:
            training_question.extend([query] * len(bm25_data))
            training_snippet.extend([snippets[bm25_data[i]] for i in range(len(bm25_data))])
    if is_train:
        dictions = {'question': training_question, 'start_idx': training_start_idx, 'end_idx': trainging_end_idx, 'snippet': training_snippet, 'ans': traing_ans}
    else:
        dictions = {'question': training_question, 'snippet': training_snippet}
    return dictions

# %%
train_dic = create_bm25_data('NLP-2-Dataset/train_BM25', train_data)    #question, ans, snippets, start_idx, end_idx
val_dic = create_bm25_data('NLP-2-Dataset/val_BM25', val_data)
test_dic = create_bm25_data('NLP-2-Dataset/test_BM25_all', test_data, False) #question, snippet

# %%
test_dic.keys()
print(len(test_dic['question']))

# memory break
time.sleep(10)
print("time sleep 10s")


# %%
from torch.utils.data import Dataset, DataLoader

class QA_Dataset(Dataset):
    def __init__(self, data_dic, tokenizer, is_train=True):
        self.dict = data_dic
        self.encodings = tokenizer(self.dict['question'],   
                                   self.dict['snippet'],   # 這邊有問題
                                   add_special_tokens=True,
                                   padding=True,     # padding 先不做-> https://zhuanlan.zhihu.com/p/414552021，但最後還是做了
                                   return_tensors='pt',
                                   return_token_type_ids=True,      # 0 = snippet, 1 = question
                                   return_offsets_mapping=True,      # mapping  回去原本的地方(token to 原本的string)
                                   truncation=True)     
        
        if is_train:
            self.start_idx = self.dict['start_idx']
            self.end_idx = self.dict['end_idx']
            self.ans = self.dict['ans']
            
        self.len = len(self.encodings['input_ids'])
       

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) if key != 'snippet' and key != 'ans' else val[idx] for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return self.len
    
    def add_token_positions(self):
        # 初始化列表以包含答案start/end的標記索引
        start_positions = []
        end_positions = []
        for i in range(self.len):
            if self.start_idx[i] == -1 or self.end_idx[i] == -1:
                start_positions.append(0)
                end_positions.append(0)
                continue
            
            # 使用char_to_token方法追加開始/結束標記位置
            start_positions.append(self.encodings.char_to_token(i, self.start_idx[i], sequence_index=1))
            end_positions.append(self.encodings.char_to_token(i, self.end_idx[i], sequence_index=1))

            # 如果起始位置為None，則答案已被截斷
            if start_positions[-1] is None:
                start_positions[-1] = 0 
            # end position無法找到，char_to_token找到了空格，所以移動位置直到找到為止
            shift = 1
            while end_positions[-1] is None and self.end_idx[i] - shift > 0:
                end_positions[-1] = self.encodings.char_to_token(i, self.end_idx[i] - shift, sequence_index=1)
                shift += 1
            if end_positions[-1] is None:
                end_positions[-1] = 0
        # 用新的基於標識的開始/結束位置更新我們的encodings對象
        self.encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

# %%
time.sleep(5)
print("time sleep 5 sec")
train_dataset = QA_Dataset(data_dic= train_dic, tokenizer= tokenizer)
val_dataset = QA_Dataset(data_dic= val_dic, tokenizer= tokenizer)
test_dataset = QA_Dataset(data_dic= test_dic, tokenizer= tokenizer, is_train=False)

# %%
train_dataset.add_token_positions()
val_dataset.add_token_positions()

# %% [markdown]
# ## Train

# %%
from torch.nn.utils import clip_grad_norm_
from transformers import get_linear_schedule_with_warmup, AdamW, get_polynomial_decay_schedule_with_warmup
import copy
from tqdm.auto import tqdm

# %%
def train(model, train_loader, optimizer, scheduler=None):
    # 訓練模型
    model.train()
    total_loss = 0
    predictions = []
    loop = tqdm(train_loader, leave=True, ncols=75)
    for (step, batch) in enumerate(loop):
        # 將所有張量移至GPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        
        mapping = batch['offset_mapping'].numpy()

        loss, start_logits, end_logits = model(input_ids, attention_mask, token_type_ids, start_positions, end_positions)    # **batch = input_ids, attention_mask, token_type_ids, start_positions, end_positions
        # loss = outputs[0]   # outputs = loss, start_logits, end_logits
        # print(end_logits.shape)
        for i in range(len(end_logits)):
            start_index = int(torch.argmax(start_logits[i]))
            end_index = int(torch.argmax(end_logits[i]))
            if start_index == 0 and end_index == 0:
                predictions.append([(start_positions[i].item(),end_positions[i].item()), (0,0)])
                continue
            if start_index > end_index or start_index >= len(mapping[i]) or end_index >= len(mapping[i]): # or mapping[i][start_index][0] >= len(contexts[i])
                predictions.append([(start_positions[i].item(),end_positions[i].item()), (0,0)])
                continue
            predictions.append([(start_positions[i].item(),end_positions[i].item()), (start_index, end_index)])
        # if step%1000 ==0:
        #     print('Epoch=%d  step=%d/%d  loss=%.5f' % (epoch, step, batch_cnt, loss))

        loss.backward()
        total_loss += loss.item()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler: scheduler.step()
        optimizer.zero_grad()
        if(step%100 == 0 and step != 0):
            print('step={}, loss= {:.4f}, total_loss= {:.4f}'.format(step, loss.item(), total_loss/(step+1)))
        loop.set_description(f'step {step}')
        loop.set_postfix(loss=loss.item())
    # 計算平均損失
    avg_train_loss = total_loss / len(train_loader)
    return avg_train_loss, predictions  # predictions = [[ans, pred], [ans, pred], ...]

def dev(model, dev_loader):
    model.eval()
    predictions = []
    total_loss = 0
    with torch.no_grad():
        loop = tqdm(dev_loader, leave=True, ncols=50)
        for (step, batch) in enumerate(loop):
            # 將所有張量移至GPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            
            mapping = batch['offset_mapping'].numpy()
            loss, start_logits, end_logits = model(input_ids, attention_mask, token_type_ids, start_positions, end_positions)    # **batch = input_ids, attention_mask, token_type_ids, start_positions, end_positions
            # loss = outputs[0]   # outputs = loss, start_logits, end_logits
            
            for i in range(len(end_logits)):
                start_idxs = torch.topk(start_logits[i], 3)[-1].detach().cpu().numpy()
                end_idxs = torch.topk(end_logits[i], 3)[-1].detach().cpu().numpy()
                prediction = []
                flag = False
                for start_index in start_idxs:
                    for end_index in end_idxs:
                        if start_idxs[0] == 0 and end_idxs[0] == 0:
                            prediction.append([(start_positions[i].item(),end_positions[i].item()), (0,0)])
                            flag = True
                            break
                        if start_index > end_index or start_index >= len(mapping[i]) or end_index >= len(mapping[i]):
                            continue
                        if start_index != 0 and end_index != 0:
                            prediction.append([(start_positions[i].item(),end_positions[i].item()), (start_index, end_index)])
                            flag = True
                            break
                    if flag:
                        break
                if len(prediction) == 0:
                    predictions.append([(start_positions[i].item(),end_positions[i].item()), (0,0)])
                else:
                    predictions.append(prediction[0])

                # start_index = int(torch.argmax(start_logits[i]))
                # end_index = int(torch.argmax(end_logits[i]))
                '''if start_index == 0 and end_index == 0:
                    predictions.append([(start_positions[i].item(),end_positions[i].item()), (0,0)])
                    continue
                if start_index > end_index or start_index >= len(mapping[i]) or end_index >= len(mapping[i]): # or mapping[i][start_index][0] >= len(contexts[i])
                    predictions.append([(start_positions[i].item(),end_positions[i].item()), (0,0)])
                    continue
                predictions.append([(start_positions[i].item(),end_positions[i].item()), (start_index, end_index)])'''
                # start_pos = mapping[i][start_index][0]
                # end_pos = mapping[i][end_index][-1]
                # predictions.append([ans[i], contexts[i][start_pos : end_pos]])

            total_loss += loss.item()
            if(step%100 == 0 and step != 0):
                print('step={}, loss= {:.4f}, total_loss= {:.4f}'.format(step, loss.item(), total_loss/(step+1)))

    avg_dev_loss = total_loss / len(dev_loader)
    return avg_dev_loss, predictions    # predictions = [[ans, pred], [ans, pred], ...]

# %%
def test(model, test_loader):
    model.eval()
    predictions = []
    total_loss = 0
    with torch.no_grad():
        loop = tqdm(test_loader, leave=True, ncols=50)
        for (step, batch) in enumerate(loop):
            # 將所有張量移至GPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            
            mapping = batch['offset_mapping'].numpy()
            start_logits, end_logits = model(input_ids, attention_mask, token_type_ids)
            
            for i in range(len(end_logits)):
                start_idxs = torch.topk(start_logits[i], 3)[-1].detach().cpu().numpy()
                end_idxs = torch.topk(end_logits[i], 3)[-1].detach().cpu().numpy()
                prediction = []
                flag = False
                for start_index in start_idxs:
                    for end_index in end_idxs:
                        if start_idxs[0] == 0 and end_idxs[0] == 0:
                            flag = True
                            break
                        if start_index > end_index or start_index >= len(mapping[i]) or end_index >= len(mapping[i]):
                            continue
                        if start_index != 0 and end_index != 0:
                            prediction.append([i, start_index, end_index])
                            flag = True
                            break
                    if flag:
                        break
                
                assert len(prediction) <= 1, 'prediction length > 1'
                if len(prediction) == 1:
                    predictions.append(prediction[0])
                    break
            
            if len(prediction) == 0:
                predictions.append([i, 0, 0])
            assert len(predictions) == step+1, 'predictions length != step+1'

    assert len(predictions) == len(test_loader), 'predictions length != test_loader length'
    return predictions    # predictions = [[ans, pred], [ans, pred], ...]

# %% [markdown]
# ### Evaluation function

# %%
import re
import collections
import string

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
 
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)
 
    def white_space_fix(text):
        return " ".join(text.split())
 
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
 
    def lower(text):
        return text.lower()
 
    return white_space_fix(remove_articles(remove_punc(lower(s))))
 
def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

# %% [markdown]
# ## Main

# %%
EPOCHS = 5
MODEL_HIDDEN_DIM = 768
model = QAModel(MODEL_HIDDEN_DIM).to(device)
optimizer = AdamW(model.parameters(), lr=15e-6, correct_bias=False)  # 1e-5

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)
# scheduler = get_polynomial_decay_schedule_with_warmup(
#     optimizer,
#     num_warmup_steps=0,
#     num_training_steps=total_steps,
#     lr_end=1e-7,
#     power=1.0,
# )

# %%
best_f1 = 0
best_f1_em = 0
pred = []

for i in range(EPOCHS):
    print(f"Epoch {i + 1} / {EPOCHS}:")
    loss, pred = train(model, train_loader, optimizer, scheduler)
    exact_match, all_match, real_f1, f1, count= 0, 0 , 0, 0, 0
    embeddings = train_dataset.encodings['offset_mapping']
    
    for j, (ans, pre_ans) in enumerate(pred):
        if ans[0] == 0 and ans[1] == 0 and pre_ans[0] == 0 and pre_ans[1] == 0:
            all_match += 1
        elif ans[0] != 0 and ans[1] != 0:
            pre_1 = embeddings[j][pre_ans[0]][0]
            pre_2 = embeddings[j][pre_ans[1]][-1]
            true_ans = train_dataset.dict['ans'][j]
            pred_ans = train_dataset.dict['snippet'][j][pre_1:pre_2]

            if true_ans == pred_ans:
                exact_match += 1
                all_match += 1
            # real_f1 += compute_f1(true_ans, pred_ans)
            f1 += compute_f1(true_ans, pred_ans)
            count += 1

    f1 = format(f1 / count, ".4f")
    print("Train Loss: {}, F1: {}, EM: {:.4f}, All: {:.4f}".format(loss, f1, exact_match / count * 1.0, all_match / len(pred) * 1.0)) #, Real F1: {real_f1}
    
    
    _, pred = dev(model, val_loader)
    val_embeddings = val_dataset.encodings['offset_mapping']
    
    exact_match, all_match, real_f1, f1, count= 0, 0 , 0, 0, 0
    for j, (ans, pre_ans) in enumerate((pred)):
        if ans[0] == 0 and ans[1] == 0 and pre_ans[0] == 0 and pre_ans[1] == 0:
            all_match += 1
        elif ans[0] != 0 and ans[1] != 0:
            pre_1 = val_embeddings[j][pre_ans[0]][0]
            pre_2 = val_embeddings[j][pre_ans[1]][-1]
            true_ans = val_dataset.dict['ans'][j]
            pred_ans = val_dataset.dict['snippet'][j][pre_1:pre_2]
            if true_ans == pred_ans:
                exact_match += 1
                all_match += 1
            # real_f1 += compute_f1(true_ans, pred_ans)
            f1 += compute_f1(true_ans, pred_ans)
            count += 1

    f1 = float(format(f1 / count, ".4f"))

    if f1 > best_f1 or (f1 == best_f1 and exact_match > best_f1_em):
        best_f1 = f1
        best_f1_em = exact_match
        torch.save(model.state_dict(), "best_model.pkl")
    print("Val F1: {}, EM: {:.4f}, All: {:.4f}".format(f1, exact_match / count * 1.0, all_match / len(pred) * 1.0))     #, Real F1: {real_f1}

# %% [markdown]
# #### test

# %%
model.load_state_dict(torch.load("best_model/best_model_1e-5_warm_mistake.pkl"))
pred = test(model, test_loader)
val_embeddings = test_dataset.encodings['offset_mapping']

output_ans = []
for (j, pre_ans) in enumerate((pred)):
    if pre_ans[0] == 0 and pre_ans[1] == 0:
        output_ans.append("")
    else:
        idx = pre_ans[0]
        pre_1 = val_embeddings[16 * j + idx][pre_ans[1]][0]
        pre_2 = val_embeddings[16 * j + idx][pre_ans[2]][-1]
        pred_ans = test_dataset.dict['snippet'][16 * j + idx][pre_1:pre_2]
        output_ans.append(pred_ans)

print(len(output_ans))
with open("output_ans.txt", "w") as f:
    for ans in output_ans:
        f.write(ans + "\n")

print("---done----")