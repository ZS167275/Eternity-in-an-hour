#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from scipy.stats import entropy
from gensim.models import Word2Vec
import lightgbm as lgb
import time
import gc
import Geohash
# unix 内核的加速操作
from pandarallel import pandarallel
pandarallel.initialize()
pd.set_option('display.max_columns', None)
import os
import warnings
warnings.filterwarnings('ignore')


# In[2]:


np.random.seed(42)


# In[3]:


def reduce_mem(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('{:.2f} Mb, {:.2f} Mb ({:.2f} %)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    gc.collect()
    return df


# In[4]:


def get_ctr(data,col):
    f1 = data.groupby(col,as_index=False)['target'].agg({
        '{}_ctr_mean'.format('_'.join(col)):'mean',
    })
    f2 = data.groupby(col + ['hour'],as_index=False)['target'].agg({
        '{}_hour_ctr_mean'.format('_'.join(col)):'mean',
    })
    f = pd.merge(f1,f2,on=col,how='outer',copy=False)
    del f1,f2
    return f


# In[5]:


def get_ctr_mean(data,col):
    f1 = data.groupby(col,as_index=False)['{}_ctr_mean'.format('_'.join(col))].agg({
                                            '{}_ctr_mean_mean'.format('_'.join(col)):'mean',
                                            '{}_ctr_mean_max'.format('_'.join(col)):'max',
                                            '{}_ctr_mean_median'.format('_'.join(col)):'median',
                                            '{}_ctr_mean_min'.format('_'.join(col)):'min',
                                            '{}_ctr_mean_var'.format('_'.join(col)):'var',                             
    })
    
    f2 = data.groupby(col + ['hour'],as_index=False)['{}_hour_ctr_mean'.format('_'.join(col))].agg({
                                            '{}_hour_ctr_mean_mean'.format('_'.join(col)):'mean',
                                            '{}_hour_ctr_mean_max'.format('_'.join(col)):'max',
                                            '{}_hour_ctr_mean_median'.format('_'.join(col)):'median',
                                            '{}_hour_ctr_mean_min'.format('_'.join(col)):'min',
                                            '{}_hour_ctr_mean_var'.format('_'.join(col)):'var',
    })

    f = pd.merge(f1,f2,on=col,how='outer',copy=False)
    del f1,f2
    return f


# In[6]:


import os
def get_emb(data,f1,f2):
    tmp = data.groupby([f1], as_index=False)[f2].agg({'{}_{}_list'.format(f1, f2): list})
    sentences = tmp['{}_{}_list'.format(f1, f2)].values.tolist()
    del tmp['{}_{}_list'.format(f1, f2)]
    for i in range(len(sentences)):
        sentences[i] = [str(x) for x in sentences[i]]   
    if os.path.exists('./w2v_{}_{}.model'.format(f1, f2)):
        model = Word2Vec.load('./w2v_{}_{}.model'.format(f1, f2))
    else:
        model = Word2Vec(sentences, size=8, window=5, min_count=1, sg=1, hs=0, seed=42)
        model.save('./w2v_{}_{}.model'.format(f1, f2))
    emb_matrix = []
    for seq in sentences:
        vec = []
        for w in seq:
            if w in model:
                vec.append(model[w])
        if len(vec) > 0:
            emb_matrix.append(np.mean(vec, axis=0))
        else:
            emb_matrix.append([0] * 8)
    emb_matrix = np.array(emb_matrix)
    for i in range(8):
        tmp['{}_{}_emb_{}'.format(f1, f2, i)] = emb_matrix[:, i]
    del model, emb_matrix, sentences
    tmp = reduce_mem(tmp)
    return tmp


# In[7]:


def get_ts_feature(data,gap_list = [1],col=['deviceid']):
    for gap in gap_list:
        data['ts_{}_{}_diff_next'.format('_'.join(col),gap)] = data.groupby(col)['ts'].shift(-gap)
        data['ts_{}_{}_diff_next'.format('_'.join(col),gap)] = data['ts_{}_{}_diff_next'.format('_'.join(col),gap)] - data['ts']
        
        data['ts_{}_{}_diff_last'.format('_'.join(col),gap)] = data.groupby(col)['ts'].shift(+gap)
        data['ts_{}_{}_diff_last'.format('_'.join(col),gap)] = data['ts'] - data['ts_{}_{}_diff_last'.format('_'.join(col),gap)]
        
        data['ts_{}_{}_diff_next_count'.format('_'.join(col),gap)] = data.groupby(col)['ts_{}_{}_diff_next'.format('_'.join(col),gap)].transform('count')
        data['ts_{}_{}_diff_last_count'.format('_'.join(col),gap)] = data.groupby(col)['ts_{}_{}_diff_last'.format('_'.join(col),gap)].transform('count')
        
        data = reduce_mem(data)
    return data
def get_second_ts(data,gap_list = [1,2,3],col=['deviceid'],con_list=[1],f='next'):
    for gap in gap_list:
        for con in con_list:
            data['ts_s_{}_{}_{}_next_{}'.format(f,'_'.join(col),gap,con)] = data.groupby(col)['ts_{}_{}_diff_{}'.format('_'.join(col),con,f)].shift(-gap)
            data['ts_s_{}_{}_{}_next_{}'.format(f,'_'.join(col),gap,con)] = data['ts_s_{}_{}_{}_next_{}'.format(f,'_'.join(col),gap,con)] - data['ts_{}_{}_diff_{}'.format('_'.join(col),con,f)]  
            
            data['ts_s_{}_{}_{}_last_{}'.format(f,'_'.join(col),gap,con)] = data.groupby(col)['ts_{}_{}_diff_{}'.format('_'.join(col),con,f)].shift(+gap)
            data['ts_s_{}_{}_{}_last_{}'.format(f,'_'.join(col),gap,con)] = data['ts_{}_{}_diff_{}'.format('_'.join(col),con,f)] - data['ts_s_{}_{}_{}_last_{}'.format(f,'_'.join(col),gap,con)]
            
        data = reduce_mem(data)
    return data


# In[23]:


user = pd.read_csv('./user.csv')
user['guid'] = user['guid'].fillna('none')


# In[24]:


user['outertag'] = user['outertag'].fillna('none')
user['outertag'] = user['outertag'].astype(str)


# In[25]:


user['outertag_list'] = user['outertag'].parallel_apply(lambda x:x.split('|') if x!='none' else ':')


# In[26]:


def get_key_words(x):
    if x == ':':
        t = []
    else:
        t = [t.split(':')[0].split('_')[0] for t in x]
    return ' '.join(t)


# In[27]:


def get_key_values(x):
    try:
        if x == ':':
            t = [0.0]
        else:
            t = [t.split(':')[1] for t in x]

        return t
    except:
        return [0.0]


# In[28]:


user['outertag_words'] = user['outertag_list'].parallel_apply(get_key_words)


# In[29]:


user['outertag_values'] = user['outertag_list'].parallel_apply(get_key_values)


# In[30]:


user['tag'] = user['tag'].fillna('none')
user['tag'] = user['tag'].astype(str)
user['tag_list'] = user['tag'].parallel_apply(lambda x:x.split('|') if x!='none' else ':')


# In[31]:


user['tag_words'] = user['tag_list'].parallel_apply(get_key_words)


# In[32]:


user['tag_values'] = user['tag_list'].parallel_apply(get_key_values)


# In[33]:


user = user.drop(['outertag','tag','outertag_list','tag_list'],axis=1)


# In[34]:


def f(x):
    x = [float(t) for t in x]
    return x


# In[35]:


user['tag_values'] = user['tag_values'].apply(lambda x:f(x))
user['mean_tag_values'] = user['tag_values'].apply(lambda x:np.mean(x))
user['max_tag_values'] = user['tag_values'].parallel_apply(lambda x:max(x))
user['min_tag_values'] = user['tag_values'].parallel_apply(lambda x:min(x))


# In[36]:


if os.path.exists('./w2v_{}_{}.model'.format('user', 'tag_words')):
    model = Word2Vec.load('./w2v_{}_{}.model'.format('user', 'tag_words'))
else:
    model = Word2Vec(user['tag_words'].parallel_apply(lambda x:x.split(' ')), size=8, window=5, min_count=1, sg=1, hs=0, seed=42)
    model.save('./w2v_{}_{}.model'.format('user', 'tag_words'))


# In[37]:


model['约会']


# In[38]:


emb_matrix = []
for seq in user['tag_words'].parallel_apply(lambda x:x.split(' ')):
    vec = []
    for w in seq:
        if w in model:
            vec.append(model[w])
    if len(vec) > 0:
        emb_matrix.append(np.mean(vec, axis=0))
    else:
        emb_matrix.append([0] * 8)
emb_matrix = np.array(emb_matrix)
for i in range(8):
    user['{}_{}_emb_{}'.format('user', 'tag_words', i)] = emb_matrix[:, i]
del model, emb_matrix


# In[39]:


if os.path.exists('./w2v_{}_{}.model'.format('user', 'outertag_words')):
    model = Word2Vec.load('./w2v_{}_{}.model'.format('user', 'outertag_words'))
else:
    model = Word2Vec(user['outertag_words'].parallel_apply(lambda x:x.split(' ')), size=8, window=5, min_count=1, sg=1, hs=0, seed=42)
    model.save('./w2v_{}_{}.model'.format('user', 'outertag_words'))
emb_matrix = []
for seq in user['outertag_words'].parallel_apply(lambda x:x.split(' ')):
    vec = []
    for w in seq:
        if w in model:
            vec.append(model[w])
    if len(vec) > 0:
        emb_matrix.append(np.mean(vec, axis=0))
    else:
        emb_matrix.append([0] * 8)
emb_matrix = np.array(emb_matrix)
for i in range(8):
    user['{}_{}_emb_{}'.format('user', 'outertag_words', i)] = emb_matrix[:, i]
del model, emb_matrix


# In[40]:


user.columns


# In[41]:


user = user[['deviceid', 'guid','user_tag_words_emb_0', 'user_tag_words_emb_1', 'user_tag_words_emb_2',
       'user_tag_words_emb_3', 'user_tag_words_emb_4', 'user_tag_words_emb_5',
       'user_tag_words_emb_6', 'user_tag_words_emb_7']]


# In[42]:


user.head()


# In[43]:


print('read train and test data')
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')


# In[44]:


train['is_train'] = 1
test['is_train'] = 0


# In[45]:


data = pd.concat([train, test], axis=0, ignore_index=False)
data = data.sort_values('ts').reset_index(drop=True)
del train,test
print('finish data concat ing')


# In[46]:


data['guid'] = data['guid'].fillna('none')


# In[47]:


data = pd.merge(data,user,on=['deviceid', 'guid'],how='left',copy=False)
del user


# In[49]:


print('change data format ing ... ...')
data['date'] = pd.to_datetime(
    data['ts'].apply(lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x / 1000)))
)
data['day'] = data['date'].dt.day
data['hour'] = data['date'].dt.hour
data['minute'] = data['date'].dt.minute
del data['date']
gc.collect()


# In[50]:


del data['timestamp']
data = reduce_mem(data)


# In[51]:


print('geohash g6')
data['g6'] = data[['lat','lng']].parallel_apply(lambda x:Geohash.encode(x[0],x[1],6),axis=1)
print('geohash g7')
data['g7'] = data[['lat','lng']].parallel_apply(lambda x:Geohash.encode(x[0],x[1],7),axis=1)


# In[52]:


print('交叉转化率特征')
train_7 = data[(data['is_train']==1)&(data['day']==7)][['deviceid','g6','newsid','netmodel','target','hour']]
train_8 = data[(data['is_train']==1)&(data['day']==8)][['deviceid','g6','newsid','netmodel','target','hour']]
train_9 = data[(data['is_train']==1)&(data['day']==9)][['deviceid','g6','newsid','netmodel','target','hour']]
train_10 = data[(data['is_train']==1)&(data['day']==10)][['deviceid','g6','newsid','netmodel','target','hour']]
for col in [['deviceid'],['deviceid','netmodel'],['deviceid','g6'],['newsid']]:
    print(col)
    ctr_7 = get_ctr(train_7,col)
    ctr_8 = get_ctr(train_8,col)
    ctr_9 = get_ctr(train_9,col)
    ctr_10 = get_ctr(train_10,col)
    
    ctr_1 = pd.concat([ctr_7,ctr_8,ctr_9,ctr_10],axis=0,ignore_index=True,sort=False)
    ctr_2 = pd.concat([ctr_8,ctr_9,ctr_10],axis=0,ignore_index=True,sort=False)
    ctr_3 = pd.concat([ctr_7,ctr_9,ctr_10],axis=0,ignore_index=True,sort=False)
    ctr_4 = pd.concat([ctr_7,ctr_8,ctr_10],axis=0,ignore_index=True,sort=False)
    ctr_5 = pd.concat([ctr_7,ctr_8,ctr_9],axis=0,ignore_index=True,sort=False)
    
    ctr_1 = get_ctr_mean(ctr_1,col)
    ctr_2 = get_ctr_mean(ctr_2,col)
    ctr_3 = get_ctr_mean(ctr_3,col)
    ctr_4 = get_ctr_mean(ctr_4,col)
    ctr_5 = get_ctr_mean(ctr_5,col)
    
    ctr_1['day'] = 11
    ctr_2['day'] = 7
    ctr_3['day'] = 8
    ctr_4['day'] = 9
    ctr_5['day'] = 10
    
    ctr = pd.concat([ctr_2,ctr_3,ctr_4,ctr_5,ctr_1],axis=0,ignore_index=True,sort=False)
    ctr = reduce_mem(ctr)
    del ctr_1,ctr_2,ctr_3,ctr_4,ctr_5,ctr_7,ctr_8,ctr_9,ctr_10
    data = pd.merge(data,ctr,on=['hour','day']+col,how='left',copy=False)
del train_7,train_8,train_9,train_10


# In[53]:


data['t'] = data['ts'].parallel_apply(lambda x:x//1000)


# In[54]:


tcmp = data.groupby(['deviceid','t']).size().reset_index()
tcmp.columns = ['deviceid','t','items']
tcmp['h_d_items'] = tcmp.groupby(['deviceid'])['items'].cumsum() - tcmp['items']
tcmp = tcmp.sort_values(['t'],ascending=False)
tcmp['f_d_items'] = tcmp.groupby(['deviceid'])['items'].cumsum() - tcmp['items']
data = pd.merge(data,tcmp,on=['deviceid','t'],copy=False,how='left')
del tcmp


# In[55]:


data['netmodel'] = data['netmodel'].map({'o':-1, 'w':1, 'g4':2, 'g2':4, 'g3':3})
data['netmodel'] = data['netmodel'].astype(int)


# In[56]:


data['app_version'] = data['app_version'].parallel_apply(lambda x:''.join(x.split('.')))
data['app_version'] = data['app_version'].astype(int)


# In[57]:


data['osversion'] = data['osversion'].parallel_apply(lambda x:''.join(x.split('.')[::-1]))
data['osversion'] = data['osversion'].astype(int)


# In[58]:


for cat_f in ['device_vendor','device_version']:
    data[cat_f] = data[cat_f].parallel_apply(lambda x:str(x).lower())
    data[cat_f] = data[cat_f].astype("category")
    data[cat_f] = data[cat_f].cat.codes


# In[59]:


for cat_f in ['deviceid','guid','newsid','g6','g7']:
    data[cat_f] = data[cat_f].astype("category")
    data[cat_f] = data[cat_f].cat.codes


# In[60]:


# 类被count编码
for cat in ['app_version','device_vendor','device_version','deviceid','lat','lng','netmodel','newsid','osversion','pos','g6','g7','guid']:
    data['{}_count'.format(cat)] = data.groupby(cat)['id'].transform('count')


# In[61]:


data = reduce_mem(data)


# In[62]:


data.head()


# In[63]:


print('make embedding feature ing')
for emb in [['deviceid','newsid']]:
    print(emb)
    tmp = get_emb(data,emb[0],emb[1])
    data = pd.merge(data,tmp,on=emb[0],how='left',copy=False)
    del tmp


# In[64]:


for col in [['deviceid'],
            ['pos','deviceid'],
            ['netmodel','deviceid'],
           ]:
    print('_'.join(col),'make','feature')
    data = get_ts_feature(data,gap_list = [1,2,3],col=col)
    data = get_second_ts(data,gap_list = [1,2,3],col=col,con_list=[1],f='next')
    data = get_second_ts(data,gap_list = [1,2,3],col=col,con_list=[1],f='last')


# In[65]:


# 增加pos信息的偏移量 pos shift(-1/+1)
for col in [['deviceid'],['netmodel','deviceid']]:
    for gap in [1,2,3]:
        print(col,gap)
        data['pos_{}_{}_diff_next'.format('_'.join(col),gap)] = data.groupby(col)['pos'].shift(-gap)
        data['pos_{}_{}_diff_next'.format('_'.join(col),gap)] = data['pos_{}_{}_diff_next'.format('_'.join(col),gap)] - data['pos']
        
        data['pos_{}_{}_diff_last'.format('_'.join(col),gap)] = data.groupby(col)['pos'].shift(+gap)
        data['pos_{}_{}_diff_last'.format('_'.join(col),gap)] = data['pos'] - data['pos_{}_{}_diff_last'.format('_'.join(col),gap)]
        data = reduce_mem(data)


# In[66]:


# 增加 netmodel 信息的偏移量 pos shift(-1/+1)
for col in [['deviceid'],['pos','deviceid']]:
    for gap in [1,2,3]:
        print(col,gap)
        data['netmodel_{}_{}_diff_next'.format('_'.join(col),gap)] = data.groupby(col)['netmodel'].shift(-gap)
        data['netmodel_{}_{}_diff_next'.format('_'.join(col),gap)] = data['netmodel_{}_{}_diff_next'.format('_'.join(col),gap)] - data['netmodel']
        
        data['netmodel_{}_{}_diff_last'.format('_'.join(col),gap)] = data.groupby(col)['netmodel'].shift(+gap)
        data['netmodel_{}_{}_diff_last'.format('_'.join(col),gap)] = data['netmodel'] - data['netmodel_{}_{}_diff_last'.format('_'.join(col),gap)]
        data = reduce_mem(data)


# In[67]:


data = reduce_mem(data)


# In[68]:


data.to_pickle('./data.pkl')


# In[ ]:





# In[4]:


data = pd.read_pickle('./data.pkl')


# In[5]:


train_data = data[data['is_train']==1]
del train_data['is_train']
del train_data['id']
X_train = train_data[train_data['day'].isin([7,8,9])]
X_valid = train_data[train_data['day'].isin([10])]
del X_train['day']
del X_valid['day']
del train_data
gc.collect()


# In[6]:


X_train = reduce_mem(X_train)
X_valid = reduce_mem(X_valid)


# In[7]:


gc.collect()


# In[16]:


lgb_param = {
    'learning_rate': 0.05,
    'boosting_type': 'dart',
    'objective': 'binary',
    'metric': 'auc',
    'max_depth': -1,
    'seed':42,
    'num_leaves':512,
    'boost_from_average':'false',
#     'two_round':'true',
    'num_threads':-1,
#     'max_bin':512
#     'device': 'gpu',
    }


# In[9]:


feature = [x for x in X_train.columns if x not in ['id', 'is_train','target','day','ts','t','items', 'max_tag_values','min_tag_values','mean_tag_values',
                                                  'level',
 'personidentification',
 'followscore',
 'personalscore',
 'gender']]
target = 'target'


# In[10]:


lgb_train = lgb.Dataset(X_train[feature].values, X_train[target].values,free_raw_data=True)
del X_train


# In[11]:


xx_score = X_valid[[target]].copy()


# In[12]:


lgb_valid = lgb.Dataset(X_valid[feature].values, X_valid[target].values, reference=lgb_train,free_raw_data=True)


# In[13]:


del data


# In[17]:


lgb_model = lgb.train(lgb_param, lgb_train, num_boost_round=1500, valid_sets=[lgb_train,lgb_valid],
                      early_stopping_rounds=50,verbose_eval=25
#                       ,learning_rates=lambda iter: 0.01 if iter <=7500 else 0.015
                     )


# Training until validation scores don't improve for 50 rounds
# [25]	training's auc: 0.979075	valid_1's auc: 0.976442
# [50]	training's auc: 0.982931	valid_1's auc: 0.978983
# [75]	training's auc: 0.985128	valid_1's auc: 0.979763
# [100]	training's auc: 0.985864	valid_1's auc: 0.978559
# [125]	training's auc: 0.985951	valid_1's auc: 0.977391
# Early stopping, best iteration is:
# [79]	training's auc: 0.985358	valid_1's auc: 0.979778
# 
# 0.8082029608320163

# In[18]:


gc.collect()
del lgb_train,lgb_valid


# In[19]:


p_test = lgb_model.predict(X_valid[feature].values,num_iteration=lgb_model.best_iteration)
del X_valid


# In[20]:


xx_score['predict'] = p_test
xx_score = xx_score.sort_values('predict',ascending=False)
xx_score = xx_score.reset_index()
xx_score.loc[xx_score.index<=int(xx_score.shape[0]*0.103),'score'] = 1
xx_score['score'] = xx_score['score'].fillna(0)


# In[21]:


ux = f1_score(xx_score['target'],xx_score['score'])
print(ux)


# In[22]:


f_imp = lgb_model.feature_importance()
f_nam = feature
f_imp_df = pd.DataFrame({'f_imp':f_imp,'f_nam':f_nam})


# In[23]:


f_imp_df.sort_values(['f_imp'])


# In[24]:


f_imp_df.to_csv('./f_imp_df.csv',index=False)


# In[ ]:





# In[25]:


best_rounds = lgb_model.best_iteration


# In[26]:


best_rounds = 1000


# In[27]:


del lgb_model


# In[28]:


data = pd.read_pickle('./data.pkl')


# In[29]:


train_data = data[data['is_train']==1]


# In[30]:


X_test = data[data['is_train']==0]


# In[31]:


del data


# In[32]:


gc.collect()


# In[33]:


print(best_rounds)
lgb_train_online = lgb.Dataset(train_data[feature].values, train_data[target].values,free_raw_data=True)
del train_data


# In[ ]:


lgb_model_online = lgb.train(lgb_param, lgb_train_online, num_boost_round=best_rounds+500, valid_sets=[lgb_train_online],verbose_eval=25
#                              ,learning_rates=lambda iter: 0.01 if iter <=7500 else 0.015
                            )


# In[ ]:


X_submit = X_test[['id']].copy()
p_test_online = lgb_model_online.predict(X_test[feature].values)
del X_test


# In[ ]:


X_submit['predict'] = p_test_online
X_submit = X_submit.sort_values('predict',ascending=False)
X_submit = X_submit.reset_index()


# In[ ]:


X_submit['target'] = 0
X_submit.loc[X_submit.index<=int(X_submit.shape[0]*0.10632930998240937) + 1,'target'] = 1
# X_submit.loc[X_submit.index<=int(X_submit.shape[0]*0.103),'target'] = 1
X_submit['target'] = X_submit['target'].fillna(0)
X_submit['target'] = X_submit['target'].astype(int)
X_submit[['id','target']].to_csv('./baseline{}.csv'.format(str(ux).split('.')[1]),index=False)


# In[ ]:


X_submit.to_csv('./X_submit.csv',index=False)


# In[ ]:




