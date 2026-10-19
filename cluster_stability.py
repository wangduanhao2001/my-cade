"""Supplementary clustering checks on the unchanged 596-point six-factor reference."""
import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'support' / 'deps'))
os.environ['MPLCONFIGDIR'] = str(ROOT / 'support' / 'mplconfig')
import json, hashlib, platform, time
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from threadpoolctl import threadpool_limits

SOURCE = Path(r'F:\文章修改\正文正式RF建模\02_E2_clustering\596_positive_E2_cluster.csv')
FEATURES = ['NDVI', 'elevation', 'slope', 'river_distance', 'river_density', 'flow_accumulation']
PARAMS = dict(n_clusters=2, init='k-means++', n_init=100, max_iter=500, tol=1e-4, algorithm='lloyd')
REPEATS = 100
BOOTSTRAPS = 200
BASE_SEED = 20260919
STARTS = {'single':2026101200, 'repeated':2026101300, 'resampling':2026101400, 'bootstrap_fit':2026101500}

def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def save_csv(frame, name):
    frame.to_csv(ROOT / name, index=False, encoding='utf-8-sig', float_format='%.17g')

def fit(x, seed, n_init=100):
    settings = {**PARAMS, 'n_init':n_init, 'random_state':int(seed)}
    return KMeans(**settings).fit(x)

def align(raw, reference):
    # For K=2 evaluate both label permutations. ARI itself needs no alignment.
    direct = np.where(raw == 0, 1, 2)
    reverse = 3 - direct
    return direct if np.sum(direct == reference) >= np.sum(reverse == reference) else reverse

def describe(values):
    a = np.asarray(values, float)
    return dict(mean=float(a.mean()), sd=float(a.std(ddof=1)) if len(a)>1 else 0.,
                median=float(np.median(a)), q025=float(np.quantile(a,.025)),
                q975=float(np.quantile(a,.975)), minimum=float(a.min()), maximum=float(a.max()))

def row(model, pred, reference, index, seed, **extra):
    return dict(run=index, model_seed=seed, ARI=float(adjusted_rand_score(reference,pred)),
                agreement_fraction=float(np.mean(reference==pred)),
                changed_n=int(np.sum(reference!=pred)), Type_I_n=int(np.sum(pred==1)),
                Type_II_n=int(np.sum(pred==2)), inertia=float(model.inertia_),
                n_iter=int(model.n_iter_), **extra)

def summarize(name, frame):
    return dict(test=name, runs=len(frame), ARI=describe(frame.ARI),
                agreement_fraction=describe(frame.agreement_fraction),
                Type_I_n=describe(frame.Type_I_n), Type_II_n=describe(frame.Type_II_n),
                changed_n=describe(frame.changed_n), iteration_limit_runs=int((frame.n_iter>=500).sum()))

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    protocol = dict(source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                    sample_n=596, variables=FEATURES, reference_label='saved zone column',
                    preprocessing='Z-score, StandardScaler population SD; no log transform or clipping',
                    kmeans_parameters=PARAMS, reference_seed=BASE_SEED,
                    repeated_full_procedure_runs=REPEATS, repeated_single_start_runs=REPEATS,
                    bootstrap_runs=BOOTSTRAPS, seed_starts=STARTS,
                    bootstrap_sampling='596 rows with replacement; scaler and KMeans refitted per resample',
                    evaluation='Predict all original 596 reference points for each fit; ARI vs saved labels',
                    label_alignment='Maximum agreement across the two label permutations for counts only',
                    bootstrap_quantiles='Empirical perturbation quantiles, not predictive confidence intervals',
                    elevation_ablation='Remove elevation only; refit five-factor Z-score and KMeans at K=2',
                    limitations='Point bootstrap does not resolve spatial dependence; K fixed at 2; no RF retraining',
                    software=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
                                  scipy=scipy.__version__,sklearn=sklearn.__version__))
    dump(ROOT / 'support' / 'protocol.json', protocol)
    df = pd.read_csv(SOURCE, float_precision='round_trip')
    assert len(df)==596 and df.sample_id.is_unique and set(df.zone)=={1,2}
    x = df[FEATURES].to_numpy(float)
    assert np.isfinite(x).all()
    reference = df.zone.to_numpy(int)
    scaler = StandardScaler().fit(x)
    z = scaler.transform(x)
    start = time.time()
    with threadpool_limits(limits=1):
        baseline = fit(z, BASE_SEED)
        reproduced = align(baseline.labels_, reference)
        assert np.array_equal(reproduced, reference), 'Reference classification could not be reproduced exactly.'
        print('Reference reproduced exactly', dict(zip(*np.unique(reference,return_counts=True))),flush=True)
        tests = {}
        for title, offset, n_init in [('Repeated multistart',STARTS['repeated'],100),
                                     ('Single-start diagnostic',STARTS['single'],1)]:
            records=[]
            for i in range(REPEATS):
                seed=offset+i; m=fit(z,seed,n_init); pred=align(m.labels_,reference)
                records.append(row(m,pred,reference,i+1,seed,n_init=n_init))
                if (i+1)%25==0:print(title,i+1,'elapsed',round(time.time()-start,1),flush=True)
            tests[title]=pd.DataFrame(records)
        records=[];predictions=[]
        for i in range(BOOTSTRAPS):
            seed=STARTS['resampling']+i;model_seed=STARTS['bootstrap_fit']+i
            indices=np.random.default_rng(seed).integers(0,len(x),size=len(x))
            s=StandardScaler().fit(x[indices]); m=fit(s.transform(x[indices]),model_seed)
            pred=align(m.predict(s.transform(x)),reference);predictions.append(pred)
            records.append(row(m,pred,reference,i+1,model_seed,resampling_seed=seed,
                               unique_sample_n=int(len(np.unique(indices))),n_init=100))
            if (i+1)%25==0:print('Bootstrap',i+1,'elapsed',round(time.time()-start,1),flush=True)
        tests['Point bootstrap']=pd.DataFrame(records)
        kept=[f for f in FEATURES if f!='elevation']
        reduced_scaler=StandardScaler().fit(df[kept].to_numpy(float))
        rz=reduced_scaler.transform(df[kept].to_numpy(float))
        reduced=fit(rz,BASE_SEED); reduced_labels=align(reduced.labels_,reference)
        ablation=row(reduced,reduced_labels,reference,1,BASE_SEED,n_init=100,
                     silhouette=float(silhouette_score(rz,reduced.labels_)))
    summaries={name:summarize(name,t) for name,t in tests.items()}
    profiles=[]
    for setting,labels in [('Reference',reference),('Without elevation',reduced_labels)]:
        for f in FEATURES:
            for group in [1,2]:
                a=df.loc[labels==group,f].to_numpy(float)
                profiles.append(dict(setting=setting,variable=f,type=group,n=len(a),
                                     mean=float(a.mean()),median=float(np.median(a)),sd=float(a.std(ddof=1))))
    profile=pd.DataFrame(profiles)
    matrix=pd.crosstab(pd.Series(reference,name='Reference type'),
                       pd.Series(reduced_labels,name='Without elevation type')).reindex(index=[1,2],columns=[1,2],fill_value=0)
    prediction_matrix=np.array(predictions,dtype=np.uint8)
    stability=df[['sample_id','longitude','latitude']].copy()
    stability['reference_type']=reference
    stability['without_elevation_type']=reduced_labels
    stability['bootstrap_agreement_fraction']=(prediction_matrix==reference).mean(axis=0)
    save_csv(tests['Repeated multistart'],'repeated_initialization_100.csv')
    save_csv(tests['Single-start diagnostic'],'single_start_diagnostic_100.csv')
    save_csv(tests['Point bootstrap'],'bootstrap_200.csv')
    save_csv(profile,'environmental_profiles.csv')
    save_csv(stability,'point_stability_and_elevation_ablation.csv')
    save_csv(matrix.rename_axis(index='reference_type',columns='without_elevation_type').reset_index(),
             'elevation_ablation_transition.csv')
    np.savez_compressed(ROOT/'support'/'bootstrap_reference_predictions.npz',labels=prediction_matrix,
                        sample_id=df.sample_id.to_numpy(str),reference_labels=reference)
    flat=[]
    for name,s in summaries.items():
        flat.append(dict(test=name,runs=s['runs'],ARI_mean=s['ARI']['mean'],ARI_sd=s['ARI']['sd'],
                         ARI_median=s['ARI']['median'],ARI_q025=s['ARI']['q025'],ARI_q975=s['ARI']['q975'],
                         ARI_min=s['ARI']['minimum'],ARI_max=s['ARI']['maximum'],
                         Type_I_min=s['Type_I_n']['minimum'],Type_I_max=s['Type_I_n']['maximum'],
                         Type_II_min=s['Type_II_n']['minimum'],Type_II_max=s['Type_II_n']['maximum']))
    flat.append(dict(test='Without elevation',runs=1,ARI_mean=ablation['ARI'],
                     Type_I_min=ablation['Type_I_n'],Type_I_max=ablation['Type_I_n'],
                     Type_II_min=ablation['Type_II_n'],Type_II_max=ablation['Type_II_n']))
    save_csv(pd.DataFrame(flat),'summary.csv')
    results=dict(protocol=protocol,reference_counts={str(k):int(v) for k,v in zip(*np.unique(reference,return_counts=True))},
                 reference_reproduction_ARI=float(adjusted_rand_score(reference,reproduced)),
                 reference_inertia=float(baseline.inertia_),reference_n_iter=int(baseline.n_iter_),
                 summaries=summaries,elevation_ablation=ablation,transition=matrix.to_numpy().tolist(),
                 elapsed_seconds=time.time()-start)
    dump(ROOT/'results.json',results)
    # Publication outputs use standard plotting libraries, not raster image generation.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,3,figsize=(12.0,3.5),gridspec_kw={'width_ratios':[1.1,1,1.3]})
    values=[tests['Repeated multistart'].ARI.to_numpy(),tests['Point bootstrap'].ARI.to_numpy()]
    axes[0].boxplot(values,tick_labels=['Initialization','Bootstrap'],showfliers=False,
                    widths=.45,medianprops={'color':'#333333'})
    for i,a in enumerate(values,1):
        jitter=np.random.default_rng(19+i).uniform(-.12,.12,len(a))
        axes[0].scatter(i+jitter,a,s=7,alpha=.35,color=['#2878A0','#E38B25'][i-1],edgecolors='none')
    axes[0].set(ylim=(-.04,1.07),ylabel='Adjusted Rand index',title='(a) Agreement with reference')
    axes[0].grid(axis='y',alpha=.18)
    im=axes[1].imshow(matrix.to_numpy(),cmap='Blues')
    for i in range(2):
        for j in range(2):
            val=int(matrix.iloc[i,j]);axes[1].text(j,i,str(val),ha='center',va='center',
                                                   color='white' if val>matrix.to_numpy().max()/2 else 'black')
    axes[1].set(xticks=[0,1],xticklabels=['Type I','Type II'],yticks=[0,1],yticklabels=['Type I','Type II'],
                xlabel='Classification without elevation',ylabel='Reference classification',
                title='(b) Elevation ablation')
    gaps=[]
    for f in kept:
        ix=FEATURES.index(f)
        gaps.append((df.loc[reduced_labels==2,f].mean()-df.loc[reduced_labels==1,f].mean())/scaler.scale_[ix])
    labels={'NDVI':'NDVI','slope':'Slope','river_distance':'Distance to rivers',
            'river_density':'Drainage density','flow_accumulation':'Flow accumulation'}
    axes[2].barh([labels[f] for f in kept],gaps,color=['#E38B25' if g>=0 else '#2878A0' for g in gaps])
    axes[2].axvline(0,color='#333333',lw=.8)
    axes[2].invert_yaxis()
    axes[2].set(xlabel='Mean difference (Type II − Type I) / reference SD',
                title='(c) Contrasts without elevation')
    axes[2].grid(axis='x',alpha=.18)
    fig.tight_layout(w_pad=2)
    for ext in ['png','pdf','svg']:fig.savefig(ROOT/f'clustering_stability_and_elevation_ablation.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    print('FINAL',json.dumps(results,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
