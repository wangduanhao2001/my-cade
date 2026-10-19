"""Apply the three initial-training models to the common nine-raster mapping domain."""
import json
import numpy as np
import pandas as pd
import joblib
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from settings import BASE, MASTER, FEATURES, E2, JOBS, sha, check_prepared, dump

def main():
    metadata=check_prepared()
    training=json.loads((BASE/'12_logs/rf_training_results.json').read_text(encoding='utf-8'))
    if training['master_sha256']!=sha(MASTER):raise ValueError('Models were fitted to different samples.')
    out=BASE/'13_susceptibility_maps';out.mkdir(parents=True,exist_ok=True)
    with rasterio.open(BASE/'03_type_zone/Type_zone_E2.tif') as src:
        zones=src.read(1);profile=src.profile.copy();bounds=src.bounds
    with rasterio.open(BASE/'03_type_zone/formal_RF_valid_mask.tif') as src:
        mask=src.read(1).astype(bool)
    cache=np.load(BASE/'12_logs/spatial_sampling_cache.npz')
    if list(cache['feature_names'])!=FEATURES:raise ValueError('Unexpected raster factor order.')
    ids,values=cache['cell_id'],cache['values']
    assert len(ids)==mask.sum() and np.isfinite(values).all()
    assert np.array_equal(zones.ravel()[ids],cache['zone'])
    fits={key:joblib.load(BASE/'08_models'/f'{key}_model.pkl') for key in ['Unified_RF','TypeI_RF','TypeII_RF']}
    for model in fits.values():
        if list(model.feature_names_in_)!=FEATURES:raise ValueError('Old or incompatible model: nine input columns required.')
        model.named_steps['RF'].n_jobs=JOBS
    result={k:np.full(zones.shape,-9999.,dtype=np.float32) for k in [*fits,'Partitioned_RF']}
    for start in range(0,len(ids),50000):
        ix=ids[start:start+50000];x=pd.DataFrame(values[start:start+50000],columns=FEATURES)
        z=zones.ravel()[ix]
        for key,typ in [('Unified_RF',None),('TypeI_RF',1),('TypeII_RF',2)]:
            sel=np.ones(len(x),bool) if typ is None else z==typ
            if sel.any():
                model=fits[key];p=model.predict_proba(x.loc[sel])[:,list(model.classes_).index(1)]
                result[key].ravel()[ix[sel]]=p
                if typ is not None:result['Partitioned_RF'].ravel()[ix[sel]]=p
    profile.update(driver='GTiff',dtype='float32',count=1,nodata=-9999,compress='deflate',predictor=3)
    rows=[]
    for key,a in result.items():
        valid=a!=-9999
        expected=mask if key in ['Unified_RF','Partitioned_RF'] else mask&(zones==(1 if key=='TypeI_RF' else 2))
        assert np.array_equal(valid,expected) and np.isfinite(a[valid]).all() and ((a[valid]>=0)&(a[valid]<=1)).all()
        with rasterio.open(out/f'{key}_susceptibility.tif','w',**profile) as dst:
            dst.write(a,1)
            dst.update_tags(score='Relative susceptibility; not calibrated event probability',
                            predictors='2024 environmental rasters plus long-term mean annual precipitation',
                            model='Initial training-set model; not refitted on all samples')
        rows.append({'model':key,'valid_cells':int(valid.sum()),'minimum':float(a[valid].min()),'maximum':float(a[valid].max())})
    fig,axes=plt.subplots(2,2,figsize=(11,10),constrained_layout=True)
    for ax,(key,a) in zip(axes.flat,result.items()):
        im=ax.imshow(np.ma.masked_equal(a,-9999),extent=[bounds.left,bounds.right,bounds.bottom,bounds.top],
                     origin='upper',cmap='YlOrRd',vmin=0,vmax=1,interpolation='nearest')
        ax.set_title(key.replace('_',' '));ax.set_xlabel('Easting (m)');ax.set_ylabel('Northing (m)')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Susceptibility score',shrink=.7)
    fig.savefig(out/'susceptibility_maps.png',dpi=300)
    fig.savefig(out/'susceptibility_maps.pdf');plt.close(fig)
    pd.DataFrame(rows).to_csv(out/'map_summary.csv',index=False)
    dump(out/'mapping_manifest.json',{'nine_features':FEATURES,'sample_metadata':metadata['input_time_description'],
         'source_model_hashes':{k:sha(BASE/'08_models'/f'{k}_model.pkl') for k in fits},
         'full_sample_zoning_model':str(BASE/'02_E2_clustering/E2_model.pkl'),
         'outputs':'Continuous GeoTIFF scores, plus PNG/PDF overview. Types are masked outside their zones.'})
    print('Mapping complete:',out)

if __name__=='__main__':main()
