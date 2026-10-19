"""596 unique sites; 2024 covariates; E2 fit once and spatial prediction."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import sys,json,hashlib,shutil,platform
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(r'F:\文章修改');OUT=ROOT/'正文正式RF建模'
sys.path.append(str(ROOT/'伪缺失敏感性分析/Code/pydeps'))
import numpy as np,pandas as pd,rasterio,openpyxl,joblib,sklearn,shapefile
from pyproj import Transformer,Geod
from rasterio.windows import Window,from_bounds
from rasterio.warp import transform_bounds
from rasterio.features import geometry_mask,shapes
from shapely.geometry import shape,Point,mapping
from shapely.ops import transform as transform_geom
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score,calinski_harabasz_score,davies_bouldin_score
from threadpoolctl import threadpool_limits
from run_formal_rf import FEATURES,save_csv,dump,validate_master

FACTORS={k:ROOT/'因子'/f'2024{cn}.tif' for k,cn in [('elevation','高程'),('slope','坡度'),('river_distance','河网距离'),('river_density','河网密度'),('flow_accumulation','汇流累积量'),('soil_type','土壤类型')]}
FACTORS.update(NDVI=ROOT/'NDVI/NDVI_2024.tif',CLCD=ROOT/'CLCD/CLCD_v01_2024_albert.tif')
CLUSTER=['NDVI','elevation','slope','river_distance','river_density','flow_accumulation']
SOURCE=Path(r'D:\下载\E2_空间分布与不同类型山洪点表.xlsx')
BOUNDARY=ROOT/'outputs/01a0b789-9a73-7001-835c-35f5c8dad630/support/chongqing_boundary.geojson'
SAMPLING_SEED=202601

def native_values(src,arr,r0,c0,x,y,source_crs):
    tx,ty=Transformer.from_crs(source_crs,src.crs,always_xy=True).transform(x,y)
    rr,cc=rasterio.transform.rowcol(src.transform,tx,ty)
    rr=np.asarray(rr)-r0;cc=np.asarray(cc)-c0
    valid=(rr>=0)&(cc>=0)&(rr<arr.shape[0])&(cc<arr.shape[1])
    out=np.full(len(rr),np.nan)
    v=arr[rr[valid],cc[valid]]
    out[valid]=np.where(np.ma.getmaskarray(v),np.nan,v.data)*src.scales[0]+src.offsets[0]
    out[~np.isfinite(out)]=np.nan
    return out

def main():
    for folder in ['01_data_check','02_E2_clustering','03_type_zone','04_main_pseudoabsence','05_master_dataset','12_logs']:
        (OUT/folder).mkdir(parents=True,exist_ok=True)
    wb=openpyxl.load_workbook(SOURCE,read_only=True,data_only=True)
    rows=list(wb['全部山洪点_E2'].values);wb.close()
    original=pd.DataFrame([r for r in rows[3:] if r and isinstance(r[0],(int,float))],columns=rows[2])
    assert len(original)==601
    ordered=original.sort_values('源表行号').reset_index(drop=True)
    kept=ordered.drop_duplicates(['x','y'],keep='first').copy()
    assert len(kept)==596
    old596=pd.read_csv(ROOT/'伪缺失敏感性分析_596点/Data/positive_596_factors_2024.csv')
    assert set(kept['源表行号'])==set(old596.source_row), 'Deduplication differs from approved Supplement'
    mapping_df=ordered[['源表行号','FTIME','x','y','类型简称']].merge(kept[['源表行号','x','y']].rename(columns={'源表行号':'retained_source_row'}),on=['x','y'],validate='many_to_one')
    mapping_df['retained']=mapping_df['源表行号']==mapping_df.retained_source_row
    save_csv(mapping_df,OUT/'01_data_check/positive_deduplication_mapping_601_to_596.csv')
    pos=pd.DataFrame({'sample_id':['P%04d'%v for v in kept['源表行号']],
        'source_row':kept['源表行号'].to_numpy(),'year':kept.FTIME.to_numpy(int),
        'longitude':kept.x.to_numpy(float),'latitude':kept.y.to_numpy(float),
        'old_type':kept['类型简称'].to_numpy(),'sample_type':'positive','label':1})
    with rasterio.open(FACTORS['elevation']) as src:
        crs=src.crs;aff=src.transform;gridshape=src.shape;bounds=src.bounds;profile=src.profile.copy()
    tr=Transformer.from_crs(4326,crs,always_xy=True);inv=Transformer.from_crs(crs,4326,always_xy=True)
    px,py=tr.transform(pos.longitude.to_numpy(),pos.latitude.to_numpy());pos['X']=px;pos['Y']=py
    rr,cc=rasterio.transform.rowcol(aff,px,py);rr=np.array(rr);cc=np.array(cc)
    assert ((rr>=0)&(cc>=0)&(rr<gridshape[0])&(cc<gridshape[1])).all()
    pos['grid_row']=rr;pos['grid_col']=cc
    pos['cell_id']=rr*gridshape[1]+cc
    pos['coordinate_group_id']=['PLOC_%04d'%(i+1) for i in range(len(pos))]
    boundary_data=json.loads(BOUNDARY.read_text(encoding='utf-8'))
    polygon=shape(boundary_data['features'][0]['geometry'])
    assert all(polygon.covers(Point(x,y)) for x,y in zip(pos.longitude,pos.latitude))
    projected=transform_geom(tr.transform,polygon)
    admin=geometry_mask([mapping(projected)],out_shape=gridshape,transform=aff,invert=True,all_touched=False)
    # Include reference cells for all positive locations when extracting, even at administrative edge.
    target=admin.copy();target[rr,cc]=True
    cellids=np.flatnonzero(target);gr,gc=np.unravel_index(cellids,gridshape)
    xs,ys=rasterio.transform.xy(aff,gr,gc);xs=np.asarray(xs);ys=np.asarray(ys)
    lookup=np.full(np.prod(gridshape),-1,dtype=np.int32);lookup[cellids]=np.arange(len(cellids))
    positive_index=lookup[pos.cell_id.to_numpy()]
    vals={};inventory=[];native_point_comparison=[]
    for name,path in FACTORS.items():
        print('Aligning',name,flush=True)
        with rasterio.open(path) as src:
            b=transform_bounds(crs,src.crs,*bounds,densify_pts=41)
            w=from_bounds(*b,transform=src.transform)
            c0=max(0,int(np.floor(w.col_off))-3);r0=max(0,int(np.floor(w.row_off))-3)
            c1=min(src.width,int(np.ceil(w.col_off+w.width))+3);r1=min(src.height,int(np.ceil(w.row_off+w.height))+3)
            arr=src.read(1,window=Window(c0,r0,c1-c0,r1-r0),masked=True)
            values=native_values(src,arr,r0,c0,xs,ys,crs)
            exact=native_values(src,arr,r0,c0,px,py,crs)
            vals[name]=values
            pos[name]=values[positive_index]
            assert np.isfinite(pos[name]).all(), name
            native_point_comparison.append({'variable':name,'changed_vs_native_exact_location':int((~np.isclose(pos[name],exact,rtol=1e-10,atol=1e-10)).sum()),'max_abs_change':float(np.nanmax(np.abs(pos[name]-exact)))})
            inventory.append({'variable':name,'source':str(path),'source_crs':str(src.crs),'source_resolution':list(src.res),'reference_crs':str(crs),'reference_resolution':[aff.a,abs(aff.e)],'scale':src.scales[0],'offset':src.offsets[0],'source_nodata':src.nodata,'source_size':path.stat().st_size,'source_mtime_ns':path.stat().st_mtime_ns,'valid_positive':int(np.isfinite(pos[name]).sum())})
        del arr
    save_csv(pd.DataFrame(inventory),OUT/'01_data_check/formal_factor_inventory_596_2024.csv')
    save_csv(pd.DataFrame(native_point_comparison),OUT/'01_data_check/grid_alignment_effect.csv')
    R=pos[CLUSTER].to_numpy(float)
    assert ((R[:,0]>=-1)&(R[:,0]<=1)).all() and ((R[:,2]>=0)&(R[:,2]<=90)).all() and (R[:,3:]>=0).all()
    e2=Pipeline([('scaler',StandardScaler()),('kmeans',KMeans(n_clusters=2,init='k-means++',n_init=100,random_state=20260919,max_iter=500,tol=1e-4,algorithm='lloyd'))])
    with threadpool_limits(limits=1):e2.fit(R)
    scaler=e2.named_steps['scaler'];km=e2.named_steps['kmeans'];raw_labels=km.labels_
    raw_centers=scaler.inverse_transform(km.cluster_centers_)
    terrain_class=int(np.argmax(raw_centers[:,CLUSTER.index('elevation')]))
    assert terrain_class==int(np.argmax(raw_centers[:,CLUSTER.index('slope')])), 'Elevation/slope mechanism mapping ambiguous'
    river_class=1-terrain_class;cluster_to_zone=np.zeros(2,dtype=np.uint8)
    cluster_to_zone[river_class]=1;cluster_to_zone[terrain_class]=2
    signs={'elevation':raw_centers[terrain_class,1]>raw_centers[river_class,1],
        'slope':raw_centers[terrain_class,2]>raw_centers[river_class,2],
        'NDVI':raw_centers[terrain_class,0]>raw_centers[river_class,0],
        'river_distance':raw_centers[terrain_class,3]>raw_centers[river_class,3],
        'river_density':raw_centers[river_class,4]>raw_centers[terrain_class,4],
        'flow_accumulation':raw_centers[river_class,5]>raw_centers[terrain_class,5]}
    pos['cluster_id']=raw_labels;pos['zone']=cluster_to_zone[raw_labels]
    # Keep an ordinary sklearn Pipeline so future predict requires no project-specific class.
    e2.feature_order_=CLUSTER;e2.cluster_to_zone_=cluster_to_zone;e2.reference_grid_={'crs':str(crs),'transform':list(aff),'shape':gridshape}
    joblib.dump(e2,OUT/'02_E2_clustering/E2_model.pkl',compress=3)
    save_csv(pd.DataFrame({'variable':CLUSTER,'mean':scaler.mean_,'scale':scaler.scale_,'variance':scaler.var_,'n_samples':596}),OUT/'02_E2_clustering/E2_scaler_parameters.csv')
    centers=[];stats=[]
    for c in range(2):
        z=int(cluster_to_zone[c]);subset=pos.loc[pos.zone.eq(z)]
        for j,f in enumerate(CLUSTER):
            centers.append({'cluster_id':c,'zone':z,'type_label':'Type I' if z==1 else 'Type II','variable':f,'standardized_center':float(km.cluster_centers_[c,j]),'raw_center':float(raw_centers[c,j])})
            stats.append({'zone':z,'variable':f,'n':len(subset),'mean':float(subset[f].mean()),'median':float(subset[f].median())})
    save_csv(pd.DataFrame(centers),OUT/'02_E2_clustering/E2_cluster_centers.csv')
    save_csv(pd.DataFrame(stats),OUT/'02_E2_clustering/E2_type_profiles.csv')
    save_csv(pos,OUT/'02_E2_clustering/596_positive_E2_cluster.csv')
    save_csv(pd.crosstab(pos.old_type,pos.zone).reset_index(),OUT/'02_E2_clustering/old_vs_new_types.csv')
    X=scaler.transform(R)
    cluster_report={'n':596,'counts':{str(k):int(v) for k,v in pos.zone.value_counts().sort_index().items()},'feature_order':CLUSTER,'cluster_to_zone':cluster_to_zone.tolist(),'mapping_expected_directions':{k:bool(v) for k,v in signs.items()},'silhouette':float(silhouette_score(X,raw_labels)),'CH':float(calinski_harabasz_score(X,raw_labels)),'DB':float(davies_bouldin_score(X,raw_labels)),'n_iter':int(km.n_iter_),'kmeans_params':km.get_params(),'n_unique_reference_cells':int(pos.cell_id.nunique())}
    print('E2',cluster_report,flush=True)
    six=np.column_stack([vals[n] for n in CLUSTER]);allvals=np.column_stack([vals[n] for n in FEATURES])
    in_admin=admin.ravel()[cellids]
    valid6=in_admin&np.isfinite(six).all(axis=1)
    valid8=in_admin&np.isfinite(allvals).all(axis=1)
    zone_values=np.zeros(len(cellids),np.uint8)
    for start in range(0,len(cellids),100000):
        ii=np.flatnonzero(valid6[start:start+100000])+start
        if len(ii):zone_values[ii]=cluster_to_zone[e2.predict(six[ii])]
    zone_grid=np.zeros(gridshape,np.uint8);zone_grid.ravel()[cellids]=zone_values
    assert np.array_equal(zone_grid[rr,cc],pos.zone), 'Positive spatial zones differ from fitted E2'
    profile.update(driver='GTiff',dtype='uint8',count=1,nodata=0,compress='deflate')
    with rasterio.open(OUT/'03_type_zone/Type_zone_E2.tif','w',**profile) as dst:
        dst.write(zone_grid,1);dst.update_tags(Type_1='river_network_confluence',Type_2='mountain_hillslope',n_training_positive='596',NDVI_year='2024',method='positive-fitted E2 predict only')
    with rasterio.open(OUT/'03_type_zone/formal_RF_valid_mask.tif','w',**profile) as dst:
        m=np.zeros(gridshape,np.uint8);m.ravel()[cellids]=valid8;dst.write(m,1)
    shutil.copy2(BOUNDARY,OUT/'03_type_zone/Chongqing_GADM_boundary.geojson')
    # Entire known inventory (604 rows) protects sampling even where a record is not in the chosen 596-site training set.
    knownwb=openpyxl.load_workbook(ROOT/'山洪灾害点2000-2015年.xlsx',read_only=True,data_only=True)
    known=np.array([r[1:3] for r in list(knownwb.worksheets[0].values)[1:] if r and r[1] is not None and r[2] is not None],float);knownwb.close()
    known=np.unique(np.vstack([known,pos[['longitude','latitude']].to_numpy()]),axis=0)
    kx,ky=tr.transform(known[:,0],known[:,1]);tree=cKDTree(np.c_[kx,ky])
    distance,_=tree.query(np.c_[xs,ys],workers=4)
    eligible=valid8&(distance>=500)
    geod=Geod(ellps='WGS84')
    close=np.flatnonzero(eligible&(distance<510))
    if len(close):
        lon,lat=inv.transform(xs[close],ys[close]);near=tree.query(np.c_[xs[close],ys[close]],k=8)[1]
        gd=np.vstack([geod.inv(lon,lat,known[near[:,j],0],known[near[:,j],1])[2] for j in range(8)])
        eligible[close]&=gd.min(axis=0)>=500
    rng=np.random.default_rng(SAMPLING_SEED);selected=[];areas=[]
    for zone in [1,2]:
        candidates=np.flatnonzero(eligible&(zone_values==zone));n=int(pos.zone.eq(zone).sum())
        assert len(candidates)>=n
        selected.extend(rng.choice(candidates,size=n,replace=False).tolist())
        n6=int((zone_values==zone).sum());n8=int((valid8&(zone_values==zone)).sum())
        areas.append({'zone':zone,'Type':'Type I' if zone==1 else 'Type II','six_factor_valid_cells':n6,'area_km2':n6*abs(aff.a*aff.e)/1e6,'area_fraction':n6/int(valid6.sum()),'RF_valid_cells':n8,'RF_valid_area_km2':n8*abs(aff.a*aff.e)/1e6,'RF_valid_area_fraction':n8/int(valid8.sum()),'eligible_candidate_cells':len(candidates),'positive_n':n,'pseudoabsence_n':n})
    selected=np.array(selected)
    lon,lat=inv.transform(xs[selected],ys[selected])
    negative=pd.DataFrame({'sample_id':['A%04d'%(i+1) for i in range(len(selected))],
        'coordinate_group_id':['ALOC_%04d'%(i+1) for i in range(len(selected))],
        'X':xs[selected],'Y':ys[selected],'longitude':lon,'latitude':lat,'label':0,
        'zone':zone_values[selected],'sample_type':'pseudo_absence','cell_id':cellids[selected],
        'grid_row':gr[selected],'grid_col':gc[selected],'sampling_seed':SAMPLING_SEED,'min_distance_projected_m':distance[selected]})
    for j,f in enumerate(FEATURES):negative[f]=allvals[selected,j]
    # Verify the geodesic nearest distance against ALL known locations for every selected point.
    gd=[]
    for x,y in zip(lon,lat):
        gd.append(float(geod.inv(np.full(len(known),x),np.full(len(known),y),known[:,0],known[:,1])[2].min()))
    negative['min_distance_geodesic_m']=gd
    assert min(gd)>=500 and all(polygon.covers(Point(x,y)) for x,y in zip(lon,lat))
    assert not negative.duplicated(['X','Y']).any()
    with rasterio.open(OUT/'03_type_zone/Type_zone_E2.tif') as src:
        assert np.array_equal(np.array([v[0] for v in src.sample(zip(negative.X,negative.Y))]),negative.zone)
    save_csv(negative,OUT/'04_main_pseudoabsence/pseudoabsence_596_seed_202601.csv')
    master=pd.concat([pos,negative],ignore_index=True)
    primary=['sample_id','coordinate_group_id','X','Y','label','zone','sample_type',*FEATURES]
    master=master[primary+[c for c in master if c not in primary]]
    validate_master(master)
    save_csv(master,OUT/'05_master_dataset/Master_RF_dataset_1to1.csv')
    save_csv(pd.DataFrame(areas),OUT/'03_type_zone/Type_zone_area_statistics.csv')
    np.savez_compressed(OUT/'12_logs/spatial_sampling_cache.npz',cell_id=cellids,eligible=eligible,zone=zone_values,values=allvals,feature_names=np.array(FEATURES))
    # Polygonization is optional only if it becomes too large; preserve exact pixel boundaries.
    polygons=[];vector_status='written'
    for geom,z in shapes(zone_grid,mask=zone_grid>0,transform=aff):
        polygons.append((geom,int(z)))
        if len(polygons)>100000:
            vector_status='skipped: more than 100000 polygons';break
    if vector_status=='written':
        vector=OUT/'03_type_zone/Type_zone_E2.shp'
        with shapefile.Writer(str(vector),shapeType=shapefile.POLYGON,encoding='utf-8') as writer:
            writer.field('zone','N');writer.field('type_name','C',12)
            for geom,z in polygons:writer.shape(geom);writer.record(z,'Type I' if z==1 else 'Type II')
        vector.with_suffix('.prj').write_text(crs.to_wkt(version='WKT1_ESRI'),encoding='utf-8');vector.with_suffix('.cpg').write_text('UTF-8')
    report={'n_original':601,'n_positive':596,'n_pseudoabsence':596,'n_master':1192,'n_unique_positive':596,
        'years':[int(pos.year.min()),int(pos.year.max())],'n_2000':int(pos.year.eq(2000).sum()),
        'source':str(SOURCE),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'boundary':str(BOUNDARY),'boundary_sha256':hashlib.sha256(BOUNDARY.read_bytes()).hexdigest(),
        'boundary_source':'GADM 4.1 Chongqing ADM1','point_crs_assumption':'EPSG:4326, inherited from validated Supplement',
        'reference_crs':str(crs),'reference_transform':list(aff),'reference_shape':gridshape,
        'resampling':'source native pixel containing each 250m reference cell centre (nearest-neighbour alignment); positives extracted from these same aligned grids',
        'admin_cells':int(admin.sum()),'valid_six_factor_cells':int(valid6.sum()),'valid_RF_cells':int(valid8.sum()),
        'buffer_known_unique_locations':len(known),'buffer_includes_root_604_inventory':True,
        'seed':SAMPLING_SEED,'sampling_rng':'numpy.default_rng PCG64; Type I then Type II; sorted candidate cell IDs; without replacement',
        'min_negative_distance_projected_m':float(negative.min_distance_projected_m.min()),'min_negative_distance_geodesic_m':min(gd),
        'areas':areas,'clustering':cluster_report,'factor_inventory':inventory,'alignment_effect':native_point_comparison,
        'vector_status':vector_status,'polygon_count':len(polygons),'versions':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__,'rasterio':rasterio.__version__}}
    dump(OUT/'12_logs/formal_data_manifest.json',report)
    print('DATA COMPLETE',json.dumps({'counts':cluster_report['counts'],'areas':areas,'vector':vector_status,'min_distance':min(gd)},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
