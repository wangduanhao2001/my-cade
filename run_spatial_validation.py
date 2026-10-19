"""Frozen nested spatial validation; never select designs from model outcomes."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='1'
import sys,json,hashlib,time,argparse
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd,joblib,rasterio
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterSampler
from threadpoolctl import threadpool_limits
from run_formal_rf import FEATURES,SPACE,make_pipeline,metrics

from settings import BASE, MASTER, SPATIAL as OUT, SIZE, BUFFER, JOBS, SHIFTS, N_CANDIDATES, OUTER_FOLDS, INNER_FOLDS, check_prepared, sha as file_sha
E2=['NDVI','elevation','slope','river_distance','river_density','flow_accumulation']
NAMES=['Unified_RF','TypeI_RF','TypeII_RF']

def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csv(df,path):df.to_csv(path,index=False,encoding='utf-8-sig',float_format='%.17g')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read():
 d=pd.read_csv(MASTER,float_precision='round_trip')
 check_prepared()
 assert d.sample_id.is_unique and set(d.label)=={0,1}
 assert np.isfinite(d[FEATURES].to_numpy()).all()
 with rasterio.open(BASE/'03_type_zone/Type_zone_E2.tif') as src:
  x,y=rasterio.transform.xy(src.transform,d.grid_row.astype(int),d.grid_col.astype(int))
 d['cx']=x;d['cy']=y
 d['base_block']=[f'{int(a)}_{int(b)}' for a,b in np.floor(d[['cx','cy']].to_numpy()/SIZE)]
 return d
def region_model(d,k,shift,seed):
 b=np.floor((d[['cx','cy']].to_numpy()-shift)/SIZE).astype(int)
 unique,inv=np.unique(b,axis=0,return_inverse=True)
 centers=(unique+.5)*SIZE+shift
 km=KMeans(n_clusters=k,n_init=100,random_state=seed,max_iter=500).fit(centers)
 order=np.lexsort((km.cluster_centers_[:,1],km.cluster_centers_[:,0]))
 mapping=np.empty(k,int);mapping[order]=np.arange(1,k+1)
 return {'kmeans':km,'mapping':mapping,'shift':np.asarray(shift)},mapping[km.labels_][inv]
def assign_regions(model,xy):
 centers=(np.floor((xy-model['shift'])/SIZE)+.5)*SIZE+model['shift']
 return model['mapping'][model['kmeans'].predict(centers)]
def buffered(train,test):
 dist=cKDTree(test[['X','Y']].to_numpy()).query(train[['X','Y']].to_numpy())[0]
 return train.loc[dist>=BUFFER].copy(),float(dist[dist>=BUFFER].min()),train.loc[dist<BUFFER].index.to_numpy()
def fit_e2(train):
 pos=train.loc[train.label==1,E2].to_numpy()
 e=Pipeline([('scaler',StandardScaler()),('kmeans',KMeans(n_clusters=2,n_init=100,random_state=20260919,max_iter=500,tol=1e-4,algorithm='lloyd'))]).fit(pos)
 sc=e.named_steps['scaler'];centers=sc.inverse_transform(e.named_steps['kmeans'].cluster_centers_)
 order=np.lexsort((centers[:,2],centers[:,1]));mapping=np.empty(2,int);mapping[order]=[1,2]
 e.cluster_to_zone_=mapping;e.feature_order_=E2
 assert np.allclose(sc.mean_,pos.mean(axis=0),rtol=1e-12,atol=1e-10)
 audit={'positive_fit_n':len(pos),'mean':sc.mean_.tolist(),'scale':sc.scale_.tolist(),'raw_centers':centers.tolist(),'cluster_to_zone':mapping.tolist(),
  'higher_elevation_center_also_higher_slope':bool(centers[order[1],2]>centers[order[0],2])}
 return e,audit
def zones(e,d):return e.cluster_to_zone_[e.predict(d[E2].to_numpy())]
def encode(train,val):
 prep=make_pipeline().named_steps['preprocess']
 return prep,prep.fit_transform(train[FEATURES]),prep.transform(val[FEATURES])
def check_pair(train,test):
 assert set(train.cell_id).isdisjoint(test.cell_id)
 assert train.label.nunique()==2 and test.label.nunique()==2
 assert cKDTree(test[['X','Y']].to_numpy()).query(train[['X','Y']].to_numpy())[0].min()>=BUFFER

def prepare():
 for folder in ['prepared','folds','models','predictions','tables','figures','logs']:(OUT/folder).mkdir(parents=True,exist_ok=True)
 manifest=OUT/'frozen_manifest.json'
 if manifest.exists():
  m=json.loads(manifest.read_text(encoding='utf-8'));assert m['code_sha256']==sha(Path(__file__));assert m['master_sha256']==sha(MASTER) and m['protocol_sha256']==sha(OUT/'验证方案_训练前固定.md')
  print('Frozen preparation exists; hashes verified.',flush=True);return
 df=read()
 candidates=[{k.replace('RF__',''):v for k,v in p.items()} for p in ParameterSampler(SPACE,n_iter=N_CANDIDATES,random_state=73001)]
 dump(OUT/'fixed_RF_candidates.json',candidates)
 counts=[];assignments=[];plans=[]
 for repeat,shift in enumerate(SHIFTS,1):
  reg,folds=region_model(df,OUTER_FOLDS,shift,20260921+repeat)
  assert df.assign(fold=folds).groupby('cell_id').fold.nunique().max()==1
  joblib.dump(reg,OUT/'prepared'/f'repeat_{repeat}_regions.pkl')
  for fold in range(1,OUTER_FOLDS+1):
   name=f'R{repeat}_F{fold}';te=df.loc[folds==fold].copy()
   tr,dist,excluded=buffered(df.loc[folds!=fold],te);check_pair(tr,te)
   assert set(tr.base_block).isdisjoint(te.base_block) if repeat==1 else True
   e,audit=fit_e2(tr);tr['fold_zone']=zones(e,tr);te['fold_zone']=zones(e,te)
   assert tr.groupby('fold_zone').label.nunique().eq(2).all() and tr.fold_zone.nunique()==2
   innerreg,innerfolds=region_model(tr,INNER_FOLDS,shift,20261000+repeat*10+fold)
   inner=[];inner_assign=[]
   for j in range(1,INNER_FOLDS+1):
    va=tr.loc[innerfolds==j].copy();it,idist,iexc=buffered(tr.loc[innerfolds!=j],va);check_pair(it,va)
    ie,ia=fit_e2(it);it['inner_zone']=zones(ie,it);va['inner_zone']=zones(ie,va)
    arrays={}
    for model,z in [('Unified_RF',None),('TypeI_RF',1),('TypeII_RF',2)]:
     a=it if z is None else it[it.inner_zone==z];b=va if z is None else va[va.inner_zone==z]
     assert a.label.nunique()==2 and b.label.nunique()==2,f'{name} inner {j} {model} missing class'
     _,xa,xb=encode(a,b)
     arrays[model]=(xa,a.label.to_numpy(),xb,b.label.to_numpy())
     counts.append({'repeat':repeat,'outer_fold':fold,'inner_fold':j,'model':model,'n_train':len(a),'positive_train':int(a.label.sum()),'n_test':len(b),'positive_test':int(b.label.sum()),'minimum_distance_m':idist})
    inner.append({'arrays':arrays,'e2_audit':ia,'train_ids':it.sample_id.tolist(),'validation_ids':va.sample_id.tolist()})
    inner_assign.extend({'sample_id':sid,'inner_fold':j,'role':role} for role,ids in [('train',it.sample_id),('validation',va.sample_id),('buffer_excluded',tr.loc[iexc].sample_id)] for sid in ids)
   csv(pd.DataFrame(inner_assign),OUT/'folds'/f'{name}_inner_assignments.csv')
   for model,z in [('Unified_RF',None),('TypeI_RF',1),('TypeII_RF',2)]:
    a=tr if z is None else tr[tr.fold_zone==z];b=te if z is None else te[te.fold_zone==z]
    counts.append({'repeat':repeat,'outer_fold':fold,'inner_fold':0,'model':model,'n_train':len(a),'positive_train':int(a.label.sum()),'n_test':len(b),'positive_test':int(b.label.sum()),'minimum_distance_m':dist})
   assignments.extend({'repeat':repeat,'outer_fold':fold,'sample_id':sid,'role':role} for role,ids in [('train',tr.sample_id),('test',te.sample_id),('buffer_excluded',df.loc[excluded].sample_id)] for sid in ids)
   obj={'repeat':repeat,'fold':fold,'train_index':tr.index.to_numpy(),'test_index':te.index.to_numpy(),'train_zone':tr.fold_zone.to_numpy(),'test_zone':te.fold_zone.to_numpy(),'outer_e2':e,'outer_e2_audit':audit,'inner':inner,'minimum_distance_m':dist}
   joblib.dump(obj,OUT/'prepared'/f'{name}.pkl',compress=3)
   plans.append({'name':name,'repeat':repeat,'fold':fold,'n_train':len(tr),'n_test':len(te),'buffer_excluded':len(excluded),'minimum_distance_m':dist,'e2':audit})
   print('Prepared',name,'train',len(tr),'test',len(te),'buffer excluded',len(excluded),flush=True)
 csv(pd.DataFrame(counts),OUT/'folds/fold_sample_counts.csv')
 csv(pd.DataFrame(assignments),OUT/'folds/all_sample_assignments.csv')
 csv(df[['sample_id','base_block','cell_id','X','Y','label']],OUT/'folds/bootstrap_blocks.csv')
 dump(manifest,{'frozen_at':time.strftime('%Y-%m-%d %H:%M:%S'),'master_sha256':sha(MASTER),'protocol_sha256':sha(OUT/'验证方案_训练前固定.md'),'candidate_sha256':sha(OUT/'fixed_RF_candidates.json'),'code_sha256':sha(Path(__file__)), 'n_candidates':N_CANDIDATES,'n_outer_runs':len(SHIFTS)*OUTER_FOLDS,'n_inner':INNER_FOLDS,'grid_m':SIZE,'buffer_m':BUFFER,'plans':plans})
 print('PREPARATION FROZEN; no RF performance used.',flush=True)

def candidate_score(name,i,params,inner):
 from sklearn.metrics import roc_auc_score
 scores=[]
 for info in inner:
  a,y,b,v=info['arrays'][name]
  model=RandomForestClassifier(**params,bootstrap=True,random_state=73001,n_jobs=1).fit(a,y)
  prob=model.predict_proba(b)[:,list(model.classes_).index(1)]
  scores.append(float(roc_auc_score(v,prob)))
 return {'model':name,'candidate_id':i,'mean_inner_AUC':float(np.mean(scores)),**{f'inner{j+1}_AUC':v for j,v in enumerate(scores)},**params}
def positive(model,d):return model.predict_proba(d[FEATURES])[:,list(model.classes_).index(1)]
def train_one(df,name,candidates):
 completion=OUT/'folds'/f'{name}_completed.json'
 if completion.exists():print('Resume: already completed',name,flush=True);return
 obj=joblib.load(OUT/'prepared'/f'{name}.pkl');start=time.time()
 scorepath=OUT/'folds'/f'{name}_candidate_scores.csv'
 if scorepath.exists():scores=pd.read_csv(scorepath).to_dict('records');assert len(scores)==3*N_CANDIDATES
 else:
  scores=joblib.Parallel(n_jobs=JOBS,backend='loky')(joblib.delayed(candidate_score)(model,i,p,obj['inner']) for model in NAMES for i,p in enumerate(candidates))
  csv(pd.DataFrame(scores),scorepath)
 print('Inner searches done',name,round(time.time()-start,1),'seconds',flush=True)
 train=df.loc[obj['train_index']].copy();test=df.loc[obj['test_index']].copy()
 train['fold_zone']=obj['train_zone'];test['fold_zone']=obj['test_zone']
 fits={};selection=[]
 for model,z in [('Unified_RF',None),('TypeI_RF',1),('TypeII_RF',2)]:
  choice=sorted([r for r in scores if r['model']==model],key=lambda r:(-r['mean_inner_AUC'],r['candidate_id']))[0]
  params=candidates[int(choice['candidate_id'])];a=train if z is None else train[train.fold_zone==z]
  pipe=make_pipeline();pipe.set_params(**{'RF__'+k:v for k,v in params.items()});pipe.fit(a[FEATURES],a.label)
  fits[model]=pipe;selection.append({'model':model,'candidate_id':int(choice['candidate_id']),'inner_AUC':choice['mean_inner_AUC'],**params})
  joblib.dump(pipe,OUT/'models'/f'{name}_{model}.pkl',compress=3)
 joblib.dump(obj['outer_e2'],OUT/'models'/f'{name}_E2.pkl',compress=3)
 pred=test[['sample_id','label','fold_zone','base_block','cell_id','X','Y']].copy()
 pred['repeat']=obj['repeat'];pred['outer_fold']=obj['fold']
 pred['Unified_probability']=positive(fits['Unified_RF'],test)
 part=np.empty(len(test))
 for z,model in [(1,'TypeI_RF'),(2,'TypeII_RF')]:
  ix=np.flatnonzero(test.fold_zone.to_numpy()==z)
  if len(ix):part[ix]=positive(fits[model],test.iloc[ix])
 pred['Partitioned_probability']=part
 csv(pred,OUT/'predictions'/f'{name}_test_predictions.csv')
 perf=[]
 for model,col in [('Unified','Unified_probability'),('Partitioned','Partitioned_probability')]:
  for z in [0,1,2]:
   ix=np.ones(len(pred),bool) if z==0 else pred.fold_zone.to_numpy()==z
   if pred.loc[ix,'label'].nunique()<2:raise ValueError(f'{name}: outer subgroup missing label {z}')
   m=metrics(pred.loc[ix,'label'],pred.loc[ix,col],model);perf.append({'repeat':obj['repeat'],'outer_fold':obj['fold'],'zone':z,**m})
 csv(pd.DataFrame(perf),OUT/'tables'/f'{name}_metrics.csv')
 dump(completion,{'name':name,'seconds':time.time()-start,'selection':selection,'primary':{r['Model']:r['ROC-AUC'] for r in perf if r['zone']==0},'master_sha256':sha(MASTER)})
 print('OUTER COMPLETE',name,'AUC',[(r['Model'],round(r['ROC-AUC'],4)) for r in perf if r['zone']==0],'seconds',round(time.time()-start,1),flush=True)

def spatial_predictions(df):
 cache=np.load(BASE/'12_logs/spatial_sampling_cache.npz');features=list(cache['feature_names']);assert features==FEATURES
 with rasterio.open(BASE/'03_type_zone/formal_RF_valid_mask.tif') as src:
  validmask=src.read(1).astype(bool);profile=src.profile.copy();aff=src.transform;shape=src.shape
 cell=cache['cell_id'];values=cache['values'];valid=np.isfinite(values).all(axis=1)&validmask.ravel()[cell]
 cells=cell[valid];values=values[valid];rr,cc=np.unravel_index(cells,shape)
 x,y=rasterio.transform.xy(aff,rr,cc);xy=np.c_[x,y]
 assert len(cells)==int(validmask.sum())
 for repeat in range(1,len(SHIFTS)+1):
  done=OUT/'predictions'/f'R{repeat}_spatial_completed.json'
  if done.exists():continue
  reg=joblib.load(OUT/'prepared'/f'repeat_{repeat}_regions.pkl');folds=assign_regions(reg,xy)
  matrices={key:np.full(shape,-9999,dtype=np.float32) for key in ['Unified','Partitioned']};areas=[]
  for fold in range(1,OUTER_FOLDS+1):
   name=f'R{repeat}_F{fold}';ix=np.flatnonzero(folds==fold)
   fits={m:joblib.load(OUT/'models'/f'{name}_{m}.pkl') for m in NAMES};e=joblib.load(OUT/'models'/f'{name}_E2.pkl')
   for m in fits.values():m.named_steps['RF'].n_jobs=4
   for start in range(0,len(ix),50000):
    ind=ix[start:start+50000];chunk=pd.DataFrame(values[ind],columns=FEATURES)
    uv=positive(fits['Unified_RF'],chunk);z=zones(e,chunk);pv=np.empty(len(chunk))
    for zone,model in [(1,'TypeI_RF'),(2,'TypeII_RF')]:
     jj=np.flatnonzero(z==zone)
     if len(jj):pv[jj]=positive(fits[model],chunk.iloc[jj])
    matrices['Unified'].ravel()[cells[ind]]=uv;matrices['Partitioned'].ravel()[cells[ind]]=pv
   test=pd.read_csv(OUT/'predictions'/f'{name}_test_predictions.csv')
   for model in ['Unified','Partitioned']:
    preds=matrices[model].ravel()[cells[ix]];sample=matrices[model].ravel()[test.cell_id.astype(int)]
    # Mapping uses 2024 covariates; historical point scores may use event-year covariates.
    # Area capture here uses the 2024 map values at held-out occurrence cells, not historical sample scores.
    pos=sample[test.label.to_numpy()==1];sorted_values=np.sort(preds)
    for q in [.05,.1,.2]:
     cutoff=float(sorted_values[len(preds)-int(np.ceil(q*len(preds)))])
     area_n=int((preds>=cutoff).sum());count=int((pos>=cutoff).sum())
     areas.append({'repeat':repeat,'outer_fold':fold,'model':model,'target_area_fraction':q,'actual_area_fraction':area_n/len(preds),'threshold':cutoff,'test_region_pixels':len(preds),'selected_pixels':area_n,'positive_test_n':len(pos),'captured_n':count,'capture_fraction':count/len(pos),'evaluation_surface':'2024 map; historical occurrence locations'})
   print('SPATIAL',name,'pixels',len(ix),flush=True)
  profile.update(driver='GTiff',dtype='float32',nodata=-9999,compress='deflate',predictor=3,count=1)
  for model,a in matrices.items():
   with rasterio.open(OUT/'predictions'/f'R{repeat}_{model}_spatial_OOF.tif','w',**profile) as dst:
    dst.write(a,1);dst.update_tags(meaning='Out-of-spatial-region predictions; five separately fitted models mosaicked',repeat=str(repeat),buffer_m='5000')
  csv(pd.DataFrame(areas),OUT/'tables'/f'R{repeat}_area_capture.csv');dump(done,{'repeat':repeat,'valid_cells':len(cells)})

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--spatial-only',action='store_true');args=ap.parse_args()
 with threadpool_limits(limits=1):
  check_prepared()
  OUT.mkdir(parents=True,exist_ok=True)
  protocol=Path(__file__).with_name('spatial_protocol.md')
  dest=OUT/'验证方案_训练前固定.md'
  if not dest.exists():dest.write_bytes(protocol.read_bytes())
  elif dest.read_bytes()!=protocol.read_bytes():raise ValueError('Protocol changed; use fresh output directory.')
  prepare()
  if args.prepare:return
  df=read();m=json.loads((OUT/'frozen_manifest.json').read_text(encoding='utf-8'))
  assert m['candidate_sha256']==sha(OUT/'fixed_RF_candidates.json')
  candidates=json.loads((OUT/'fixed_RF_candidates.json').read_text(encoding='utf-8'))
  if not args.spatial_only:
   for plan in m['plans']:train_one(df,plan['name'],candidates)
  spatial_predictions(df)
  print('ALL TRAINING AND SPATIAL PREDICTIONS COMPLETE',flush=True)
if __name__=='__main__':main()
