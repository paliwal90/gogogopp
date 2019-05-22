from utils import *
#%matplotlib inline

#Uncomment matplotlib.use('Agg') when run in bash
'''
The final submission was blend of few different models with different k_fold seeds.
'''

K_folds = 10 #tried 7 also for couple of models.
kfold_seed=12
script_id='var_62'
logger = logging.getLogger('ok_application')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('logfile_{}.log'.format(script_id))
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

def scale(tr,te,te_real,columns):
    for col in columns:
        if col.startswith('var_'):
            mean,std = tr[col].append(te_real[col]).mean(), tr[col].append(te_real[col]).std()
            tr[col] = (tr[col]-mean)/(1.*std)
            te[col] = (te[col]-mean)/(1.*std)
    return tr,te

def scale_minus_min(tr,te,te_real, columns):
    for col in columns:
        if col.startswith('var_'):
            min_ = tr[col].append(te_real[col]).min()
            tr[col] = tr[col]-min_
            te[col] = te[col]-min_
    return tr,te

def reverse(tr,te):
    reverse_list = [0,1,2,3,4,5,6,7,8,11,15,16,18,19,
                22,24,25,26,27,41,29,
                32,35,37,40,48,49,47,
                55,51,52,53,60,61,62,103,65,66,67,69,
                70,71,74,78,79,
                82,84,89,90,91,94,95,96,97,99,
                105,106,110,111,112,118,119,125,128,
                130,133,134,135,137,138,
                140,144,145,147,151,155,157,159,
                161,162,163,164,167,168,
                170,171,173,175,176,179,
                180,181,184,185,187,189,
                190,191,195,196,199]
    reverse_list = ['var_%d'%i for i in reverse_list]
    for col in reverse_list:
        tr[col] = tr[col]*(-1.)
        te[col] = te[col]*(-1.)
    return tr,te


def freq_encoding(tr, te_real, te_fake, variables, append_suffix=True):
    for variable in variables:
        #logger.info(variable)
        df_combine = pd.concat([tr[variable], te_real[variable]], axis=0)
        counts = df_combine.value_counts().reset_index()
        #counts.loc[counts[variable]<=10,variable]=10
        z1 = dict(zip(counts['index'], counts[variable])) #converts to dictionary, key_level:count_value
        df_combine = df_combine.map(z1) 
        if append_suffix:
            tr[variable+"_freq"] = df_combine[:len(tr)]
            te_real[variable+"_freq"] = df_combine[len(tr):]
        else:
            tr[variable] = df_combine[:len(tr)]
            te_real[variable] = df_combine[len(tr):]
        del df_combine, counts, z1
        
        #te_fake
        df_combine = pd.concat([te_fake[variable]], axis=0)
        counts = df_combine.value_counts().reset_index()
        #counts.loc[counts[variable]<=10,variable]=10
        z1 = dict(zip(counts['index'], counts[variable])) #converts to dictionary, key_level:count_value
        df_combine = df_combine.map(z1) 
        if append_suffix:
            te_fake[variable+"_freq"] = df_combine
        else:
            te_fake[variable] = df_combine
        del df_combine, counts, z1

    return tr, te_real, te_fake

def mul_feats(tr, te_real, te_fake, columns):
    for col in columns:
        mean,std = tr[col].append(te_real[col]).mean(), tr[col].append(te_real[col]).std()
        tr_col = (tr[col]-mean)/(1.*std)
        te_col = (te[col]-mean)/(1.*std)
        te_real_col = (te_real[col]-mean)/(1.*std)
        te_fake_col = (te_fake[col] - te_fake[col].mean())/ (1.*te_fake[col].std())
        
        min_,mean_ = tr_col.append(te_real_col).min(),tr_col.append(te_real_col).mean()
        tr_col = tr_col-(min_+0.0001)
        te_real_col = te_real_col-(min_+0.0001)
        
        tr[col+"_mul"] = 1.*tr_col*tr[col+"_freq"]/mean_
        te_real[col+"_mul"] = 1.*te_real_col*te_real[col+"_freq"]/mean_
        
        min_,mean_ = te_fake_col.min(),te_fake_col.mean()
        te_fake_col = te_fake_col - (min_ + 0.0001)
        te_fake[col+"_mul"] = 1.*te_fake_col*te_fake[col+"_freq"]/mean_
        
    del tr_col, te_col, te_real_col, te_fake_col
    return tr,te_real,te_fake 

def get_te_fake_real_indexes(test):
    df_test = test.drop(['ID_code'], axis=1).copy()
    df_test = df_test.values

    unique_samples = []
    unique_count = np.zeros_like(df_test)
    for feature in tqdm(range(df_test.shape[1])):
        _, index_, count_ = np.unique(df_test[:, feature], return_counts=True, return_index=True)
        unique_count[index_[count_ == 1], feature] += 1

    # Samples which have unique values are real the others are fake
    real_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) > 0)[:, 0]
    synthetic_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) == 0)[:, 0]
    
    del df_test; print gc.collect()
    
    print(len(real_samples_indexes))
    print(len(synthetic_samples_indexes))
    
    return real_samples_indexes, synthetic_samples_indexes

###### START ######

logger.info("reading data")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
features = [c for c in train.columns if c not in ['ID_code', 'target']]

real_samples_indexes, synthetic_samples_indexes = get_te_fake_real_indexes(test.copy())

logger.info("revesing")
tr,te = reverse(train.copy(),test.copy())
te_real =  te.ix[real_samples_indexes].copy()
te_fake =  te.ix[synthetic_samples_indexes].copy()

logger.info("Freq Encoding")
tr, te_real, te_fake = freq_encoding(tr, te_real, te_fake, features, append_suffix=True)

logger.info("Re ordering te indexes in correct order")
te = pd.concat([te_real, te_fake], axis=0)
te.sort_index(inplace=True)

logger.info("mul_feats")
tr, te_real, te_fake =  mul_feats(tr, te_real, te_fake, features)

logger.info("Re ordering te indexes in correct order")
te = pd.concat([te_real, te_fake], axis=0)
te.sort_index(inplace=True)

logger.info(len(features))
features = [c for c in tr.columns if c not in ['ID_code', 'target']]
logger.info(len(features))

######## Param
param_default = {
    'bagging_freq': 2,
    'bagging_fraction': 1.0,
    'boost': 'gbdt',
    'feature_fraction': 0.99,
    'feature_fraction_seed':92,
    'learning_rate': 0.025,
    'max_depth': -1,
    'metric':'auc',
    'min_data_in_leaf': 80,
    'min_sum_hessian_in_leaf': 1.01,
    'num_leaves': 3,
    'num_threads': 8,
    'tree_learner': 'serial',
    'objective': 'binary', 
    'verbosity': -1,
    'max_bin':100,
    'min_gain_to_split':0.01,
    'min_data_in_bin': 30,
    'lambda_l1':5.1,
    'lambda_l2':10.1 
}

param_1 = param_default.copy()

param_2 = param_default.copy()
param_2['feature_fraction_seed']=42; param_2['learning_rate']=0.0125;

runs=zip(['ver_62_1', 'ver_62_2'], 
         [param_1, param_2])


######## gc free
del train, test; print(gc.collect())
######## Stratified K fold

target=tr['target']
tr=tr[features]

for script_id, param in runs:
    logging.info(script_id)
    
    oof = np.zeros(len(tr))
    getVal = np.zeros(len(tr))
    predictions = np.zeros(len(te))
    feature_importance_df = pd.DataFrame()
    
    skf = StratifiedKFold(n_splits=K_folds, random_state=kfold_seed, shuffle=True)

    logger.info(skf)  
    fold_= 0

    for train_index, valid_index in skf.split(tr, target):
        logger.info("Fold idx: %d", fold_ + 1)
    
        X_train, y_train = tr.iloc[train_index], target.iloc[train_index]
        X_valid, y_valid = tr.iloc[valid_index], target.iloc[valid_index]
    
        logger.info("X_train size {} and X_valid size {}".format(X_train.shape,X_valid.shape))
 
        trn_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_valid, label=y_valid)
    
        logger.info("training")
        clf = lgb.train(param, 
                        trn_data, 
                        valid_sets = [trn_data, val_data],
                        valid_names=['train','valid'],
                        num_boost_round = 100000, 
                        verbose_eval=500,
                        early_stopping_rounds = 3000)
    
        logger.info("OOF predict")
        oof[valid_index] = clf.predict(tr.iloc[valid_index], num_iteration=clf.best_iteration)
        getVal[valid_index]+= clf.predict(tr.iloc[valid_index], num_iteration=clf.best_iteration) / (1.*K_folds)
    
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = features
        fold_importance_df["importance"] = clf.feature_importance()
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
        logger.info("Te predict")
        predictions += clf.predict(te[features], num_iteration=clf.best_iteration) / (1.*K_folds)
    
        fold_+=1

    oof_cv = roc_auc_score(target, oof)
    logger.info("OOF CV score: %f", oof_cv)

    cols = (feature_importance_df[["feature", "importance"]]
            .groupby("feature")
            .mean()
            .sort_values(by="importance", ascending=False)[:1000].index)
    best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]


    plt.figure(figsize=(14,26))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance",ascending=False))
    plt.title('LightGBM Features (averaged over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances_{}.png'.format(script_id))


    logger.info('Saving the Submission File')
    sub = pd.DataFrame({"ID_code": te.ID_code.values})
    sub["target"] = predictions
    sub.to_csv('sub_{}_{}.csv'.format(oof_cv, script_id), index=False)

    getValue = pd.DataFrame(getVal)
    getValue.to_csv("valid_kfold_{}_{}.csv".format(oof_cv, script_id))



#fig, ax = plt.subplots(figsize=(10, 10))
#lgb.plot_importance(bst1, importance_type='gain',ax=ax, max_num_features=20)
