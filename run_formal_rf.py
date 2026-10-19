"""Tune/evaluate one frozen master dataset. Does not create samples or infer zones."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import argparse
import hashlib
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, precision_recall_curve, auc,
    average_precision_score, accuracy_score, precision_score, recall_score,
    f1_score, balanced_accuracy_score, confusion_matrix)
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from settings import FEATURES, BASE, MASTER, N_CANDIDATES, INITIAL_FOLDS, JOBS, check_prepared, CFG
CATEGORICAL=['soil_type','CLCD']
CONTINUOUS=[f for f in FEATURES if f not in CATEGORICAL]
SPLIT_SEED=42001
RF_SEED=73001
SEARCH_SEED=73001
CV_SEED=42001
SPACE={
    'RF__n_estimators':[80,100,120,140,160],
    'RF__max_depth':list(range(6,13)),
    'RF__min_samples_split':list(range(5,11)),
    'RF__min_samples_leaf':list(range(8,15)),
    'RF__max_features':['sqrt'],
    'RF__max_samples':[0.7],
    'RF__class_weight':['balanced_subsample'],
}

if CFG.get('search_space'):
    SPACE=CFG['search_space']

def save_csv(df,path):
    df.to_csv(path,index=False,encoding='utf-8-sig',float_format='%.17g')

def dump(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def validate_master(df):
    required=['sample_id','coordinate_group_id','X','Y','label','zone','sample_type',*FEATURES]
    if not set(required).issubset(df):
        raise ValueError('Missing master columns: '+str(set(required)-set(df.columns)))
    assert not df[required].isna().any().any(), 'Missing data'
    assert df.sample_id.is_unique
    assert set(df.label)=={0,1} and set(df.zone)=={1,2}
    assert np.isfinite(df[['X','Y',*FEATURES]].to_numpy(float)).all()
    assert (df.loc[df.label.eq(1),'sample_type']=='positive').all()
    assert (df.loc[df.label.eq(0),'sample_type']=='pseudo_absence').all()
    assert not df.loc[df.label.eq(0)].duplicated(['X','Y']).any()
    assert (df.groupby(['X','Y']).coordinate_group_id.nunique()==1).all()
    assert (df.groupby('coordinate_group_id')[['X','Y']].nunique()<=1).all().all()
    counts=df.groupby(['zone','label']).size().unstack()
    # Preserve supplied samples: reclassification may change within-zone balance.
    assert df.label.eq(0).sum() == df.label.eq(1).sum(), 'Overall 1:1 samples required'
    for c in CATEGORICAL:
        assert np.equal(df[c],np.floor(df[c])).all()

def freeze_partition(df,out):
    """One stratified split of unique coordinate groups; no performance selection."""
    path=out/'06_train_test_split'
    path.mkdir(parents=True,exist_ok=True)
    trainfile=path/'train_sample_ids.csv';testfile=path/'test_sample_ids.csv'
    if trainfile.exists() or testfile.exists():
        assert trainfile.exists() and testfile.exists(), 'Incomplete frozen split'
        trainids=pd.read_csv(trainfile).sample_id
        testids=pd.read_csv(testfile).sample_id
        assert trainids.is_unique and testids.is_unique
        assert set(trainids).isdisjoint(testids)
        assert set(trainids)|set(testids)==set(df.sample_id)
    else:
        groups=df.groupby('partition_group_id',sort=True).agg(label=('label','first'),zone=('zone','first'))
        strata=groups.label.astype(str)+'_'+groups.zone.astype(str)
        tg,vg=train_test_split(groups.index.to_numpy(),test_size=.30,random_state=SPLIT_SEED,stratify=strata)
        trainids=df.loc[df.partition_group_id.isin(tg),'sample_id']
        testids=df.loc[df.partition_group_id.isin(vg),'sample_id']
        save_csv(pd.DataFrame({'sample_id':trainids}),trainfile)
        save_csv(pd.DataFrame({'sample_id':testids}),testfile)
    split=np.where(df.sample_id.isin(testids),'test','train')
    assignment=df[['sample_id','coordinate_group_id','partition_group_id','label','zone']].copy()
    assignment['partition']=split
    assert assignment.groupby('coordinate_group_id').partition.nunique().max()==1
    assert assignment.groupby('partition_group_id').partition.nunique().max()==1
    assert assignment.groupby(['partition','zone','label']).size().size==8
    save_csv(assignment,path/'partition_assignments.csv')
    save_csv(assignment.groupby(['partition','zone','label']).size().rename('n').reset_index(),path/'partition_counts.csv')
    return split

def make_pipeline():
    prep=ColumnTransformer([
        ('continuous','passthrough',CONTINUOUS),
        ('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CATEGORICAL),
    ])
    return Pipeline([('preprocess',prep),('RF',RandomForestClassifier(
        bootstrap=True,random_state=RF_SEED,n_jobs=1))])

def tune(train,name,out,jobs):
    cv=StratifiedGroupKFold(n_splits=INITIAL_FOLDS,shuffle=True,random_state=CV_SEED)
    strata=train.label.astype(str)+'_'+train.zone.astype(str)
    folds=list(cv.split(train[FEATURES],strata,train.partition_group_id))
    assignment=[]
    for i,(a,b) in enumerate(folds):
        assert set(train.iloc[a].coordinate_group_id).isdisjoint(train.iloc[b].coordinate_group_id)
        assert set(train.iloc[a].partition_group_id).isdisjoint(train.iloc[b].partition_group_id)
        assert train.iloc[a].label.nunique()==2 and train.iloc[b].label.nunique()==2
        assignment.extend({'sample_id':sid,'validation_fold':i+1} for sid in train.iloc[b].sample_id)
    save_csv(pd.DataFrame(assignment),out/'07_RF_tuning'/f'{name}_CV_fold_assignments.csv')
    search=RandomizedSearchCV(make_pipeline(),SPACE,n_iter=N_CANDIDATES,scoring='roc_auc',
        n_jobs=jobs,cv=folds,refit=True,random_state=SEARCH_SEED,error_score='raise',return_train_score=False)
    # Encoder is fitted inside each CV fold through the Pipeline.
    # Thread workers share the verified local dependency path on Windows.
    with joblib.parallel_backend('threading'):
        search.fit(train[FEATURES],train.label)
    params={k.removeprefix('RF__'):v for k,v in search.best_params_.items()}
    params.update(random_state=RF_SEED)
    dump(out/'07_RF_tuning'/f'{name}_best_params.json',params)
    save_csv(pd.DataFrame(search.cv_results_),out/'07_RF_tuning'/f'{name}_CV_results.csv')
    joblib.dump(search.best_estimator_,out/'08_models'/f'{name}_model.pkl',compress=3)
    signature=[{k.removeprefix('RF__'):v for k,v in row.items()} for row in search.cv_results_['params']]
    return search.best_estimator_,params,float(search.best_score_),signature

def positive_probability(model,test):
    classes=list(model.classes_)
    return model.predict_proba(test[FEATURES])[:,classes.index(1)]

def metrics(label,prob,name):
    predicted=(prob>=.5).astype(int)
    precision,recall,_=precision_recall_curve(label,prob)
    tn,fp,fn,tp=confusion_matrix(label,predicted,labels=[0,1]).ravel()
    return {'Model':name,'N_test':len(label),'ROC-AUC':roc_auc_score(label,prob),
        'PR-AUC':auc(recall,precision),'Accuracy':accuracy_score(label,predicted),
        'Precision':precision_score(label,predicted,zero_division=0),
        'Recall':recall_score(label,predicted,zero_division=0),
        'F1':f1_score(label,predicted,zero_division=0),
        'Balanced Accuracy':balanced_accuracy_score(label,predicted),
        'Specificity':float(tn/(tn+fp)),'Average Precision':average_precision_score(label,prob),
        'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp)}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--master',type=Path,default=MASTER)
    ap.add_argument('--output',type=Path,default=BASE)
    ap.add_argument('--jobs',type=int,default=JOBS)
    args=ap.parse_args();out=args.output
    check_prepared()
    if args.master.resolve()!=MASTER.resolve() or out.resolve()!=BASE.resolve():
        raise ValueError("Set master_csv/output_dir in config and prepare again; arbitrary mixed outputs are disallowed.")
    for folder in ['06_train_test_split','07_RF_tuning','08_models','09_predictions','10_tables','12_logs']:
        (out/folder).mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.master);validate_master(df)
    # A shared environmental pixel must not leak across partitions, even for distinct coordinates.
    df['partition_group_id']='CELL_'+df.cell_id.astype(int).astype(str)
    # Group all samples in one raster cell even when event-year covariates differ.
    split=freeze_partition(df,out)
    train=df.loc[split=='train'].copy();test=df.loc[split=='test'].copy()
    # No model selection code receives the test data.
    fits={};parameter_rows=[];signatures=[]
    for name,title,zone in [('Unified_RF','Unified RF',None),('TypeI_RF','Type I RF',1),('TypeII_RF','Type II RF',2)]:
        subset=train if zone is None else train.loc[train.zone.eq(zone)]
        print('Tuning',title,'training n=',len(subset),flush=True)
        model,params,score,signature=tune(subset,name,out,args.jobs)
        fits[name]=model;signatures.append(signature)
        parameter_rows.append({'Model':title,**params,'CV ROC-AUC':score})
        print('Completed',title,'CV ROC-AUC=',score,flush=True)
    assert signatures[0]==signatures[1]==signatures[2], 'Search candidates differ'
    pred=test[['sample_id','coordinate_group_id','label','zone']].copy()
    pred['Unified_probability']=positive_probability(fits['Unified_RF'],test)
    pred['Unified_prediction']=(pred.Unified_probability>=.5).astype(int)
    parts=[];performance=[metrics(test.label,pred.Unified_probability,'Unified RF')]
    for zone,name,title,column in [(1,'TypeI_RF','Type I RF','TypeI_probability'),(2,'TypeII_RF','Type II RF','TypeII_probability')]:
        subset=test.loc[test.zone.eq(zone)]
        prob=positive_probability(fits[name],subset)
        partial=pd.DataFrame({'sample_id':subset.sample_id,'Partitioned_probability':prob})
        parts.append(partial)
        pred=pred.merge(pd.DataFrame({'sample_id':subset.sample_id,column:prob}),on='sample_id',how='left',validate='one_to_one')
        performance.append(metrics(subset.label,prob,title))
    combined=pd.concat(parts,ignore_index=True)
    assert combined.sample_id.is_unique and set(combined.sample_id)==set(test.sample_id)
    pred=pred.merge(combined,on='sample_id',how='left',validate='one_to_one')
    assert pred.Partitioned_probability.notna().all()
    pred['Partitioned_prediction']=(pred.Partitioned_probability>=.5).astype(int)
    performance.append(metrics(pred.label,pred.Partitioned_probability,'Partitioned RF'))
    assert performance[0]['N_test']==performance[-1]['N_test']
    save_csv(pred,out/'09_predictions/RF_test_predictions.csv')
    save_csv(pd.DataFrame(parameter_rows),out/'10_tables/Table_RF_best_parameters.csv')
    save_csv(pd.DataFrame(performance),out/'10_tables/Table_RF_model_performance.csv')
    dump(out/'12_logs/rf_training_results.json',{'parameters':parameter_rows,'performance':performance,
        'master_sha256':hashlib.sha256(args.master.read_bytes()).hexdigest(),
        'split_seed':SPLIT_SEED,'RF_seed':RF_SEED,'search_seed':SEARCH_SEED,'cv_seed':CV_SEED,
        'search_space':SPACE,'n_iter':N_CANDIDATES,'folds':INITIAL_FOLDS,'threshold':.5,
        'PR_AUC_definition':'trapezoidal PR curve area; average precision is stored separately',
        'grouping':'250m reference cell_id; all identical coordinates and all same-pixel sites remain together',
        'n_master':len(df),'n_train':len(train),'n_test':len(test)})
    print(pd.DataFrame(performance).to_string(index=False),flush=True)

if __name__=='__main__':
    main()
