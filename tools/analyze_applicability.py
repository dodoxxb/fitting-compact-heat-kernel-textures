"""Post-hoc object heterogeneity; no outcomes used to choose the geometry clusters."""
from pathlib import Path
import json,math
import numpy as np
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import silhouette_score,adjusted_rand_score
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--code-root',type=Path,default=Path(__file__).resolve().parents[1])
parser.add_argument('--input-dir',type=Path)
parser.add_argument('--output',type=Path)
args=parser.parse_args()
CODE=args.code_root.resolve();HERE=args.input_dir or CODE/'results/applicability';OUTPUT=args.output or HERE/'results.json'
OUTPUT.parent.mkdir(parents=True,exist_ok=True)
read=lambda p:json.loads(p.read_text());plan=read(HERE/'analysis_plan.json');geom=plan['geometry_features'];additional=plan['additional_descriptors'];descs={o['uid']:o for o in read(HERE/'descriptors.json')['objects'] if o['status']=='complete'}
cohorts={}
heat=read(CODE/'results/latest/heat_features.json')
rows=[]
for o in heat['objects']:
 if not o['qualified']:continue
 seeds=[x for x in o['seeds'] if x['qualified']];assert len(seeds)==2
 effects=[x['variants']['heat_features']['endpoint']['pbr_psnr']-x['variants']['fp64_diagonal']['endpoint']['pbr_psnr'] for x in seeds]
 feats=descs[o['uid']]['features'].copy();feats['control_surface_psnr']=float(np.mean([x['variants']['fp64_diagonal']['endpoint']['native_psnr'] for x in seeds]))
 rows.append(dict(uid=o['uid'],name=o['name'],dataset=o['dataset'],delta=float(np.mean(effects)),seed_effects=effects,features=feats))
cohorts['heat_features']=rows
rows=[]
for o in read(CODE/'results/final_followup/confirmation.json')['objects']:
 seeds=[s for s in o['seeds'] if s.get('qualified')]
 if len(seeds)!=3:continue
 effects=[s['arms']['guarded_reset']['endpoint']['pbr_psnr']-s['arms']['reset_only']['endpoint']['pbr_psnr'] for s in seeds]
 d=descs[o['uid']];feats=d['features'].copy();feats['control_surface_psnr']=float(np.mean([s['arms']['reset_only']['endpoint']['native_psnr'] for s in seeds]))
 rows.append(dict(uid=o['uid'],name=o['name'],dataset=d['dataset'],delta=float(np.mean(effects)),seed_effects=effects,features=feats))
cohorts['conditional_correction']=rows
assert len(cohorts['heat_features'])==19 and len(cohorts['conditional_correction'])==19

def stats(x):
 x=np.array(x);boot=np.random.default_rng(20261014).choice(x,(10000,len(x))).mean(axis=1)
 return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),positive=int((x>0).sum()),bootstrap95=np.quantile(boot,[.025,.975]).tolist())
def spearman(x,y,seed=20261014):
 x=rankdata(x);y=rankdata(y);x=(x-x.mean())/np.linalg.norm(x-x.mean());y=(y-y.mean())/np.linalg.norm(y-y.mean());rho=float(x@y)
 perms=np.array([np.random.default_rng(seed+i).permutation(y) for i in range(10000)])
 return rho,float((1+np.count_nonzero(np.abs(perms@x)>=abs(rho)-1e-12))/10001)
def predict(rows,names):
 X=np.array([[r['features'][k] for k in names] for r in rows]);y=np.array([r['delta'] for r in rows]);pred=[];base=[]
 for i in range(len(y)):
  keep=np.arange(len(y))!=i;sc=StandardScaler().fit(X[keep]);model=Ridge(alpha=10).fit(sc.transform(X[keep]),y[keep]);pred.append(float(model.predict(sc.transform(X[[i]]))[0]));base.append(float(y[keep].mean()))
 return dict(features=names,alpha=10,predictions=pred,mean_baseline=base,mae=float(np.mean(abs(y-pred))),baseline_mae=float(np.mean(abs(y-base))),rmse=float(np.sqrt(np.mean((y-pred)**2))),baseline_rmse=float(np.sqrt(np.mean((y-base)**2))),sign_accuracy=float(np.mean((np.array(pred)>0)==(y>0))),always_positive_accuracy=float(np.mean(y>0)))
results={};associations=[]
for key,rows in cohorts.items():
 rows.sort(key=lambda r:r['uid']);X=np.array([[r['features'][f] for f in geom] for r in rows]);Z=StandardScaler().fit_transform(X);km=KMeans(n_clusters=2,n_init=30,random_state=20261014).fit(Z);labels=km.labels_;y=np.array([r['delta'] for r in rows]);data=np.array([r['dataset'] for r in rows]);lower=int(np.argmin([X[labels==c,0].mean() for c in [0,1]]));labels=(labels!=lower).astype(int)
 for row,label in zip(rows,labels):row['geometry_cluster']=int(label)
 groups={str(c):dict(**stats(y[labels==c]),mean_features={f:float(X[labels==c,j].mean()) for j,f in enumerate(geom)},datasets={d:int(np.sum((labels==c)&(data==d))) for d in sorted(set(data))},objects=[r['name'] for r in rows if r['geometry_cluster']==c]) for c in [0,1]}
 aris=[]
 for i in range(len(y)):
  keep=np.arange(len(y))!=i;z=StandardScaler().fit_transform(X[keep]);lab=KMeans(n_clusters=2,n_init=30,random_state=20261014).fit_predict(z);aris.append(float(adjusted_rand_score(labels[keep],lab)))
 diff=float(y[labels==1].mean()-y[labels==0].mean());rng=np.random.default_rng(20261014);null=[]
 for _ in range(10000):
  yp=y.copy()
  for d in sorted(set(data)):yp[data==d]=rng.permutation(y[data==d])
  null.append(yp[labels==1].mean()-yp[labels==0].mean())
 cp=float((1+np.count_nonzero(np.abs(null)>=abs(diff)-1e-12))/10001)
 associations.append(dict(cohort=key,feature='geometry_cluster',statistic=diff,p=cp,test='within-dataset permutation of cluster mean difference'))
 for feature in geom+additional:
  rho,p=spearman([r['features'][feature] for r in rows],y);associations.append(dict(cohort=key,feature=feature,statistic=rho,p=p,test='two-sided Spearman permutation'))
 max_i=int(np.argmax(y));trim=np.arange(len(y))!=max_i
 results[key]=dict(overall=stats(y),objects=rows,groups=groups,silhouette=float(silhouette_score(Z,labels)),loo_cluster_ari_median=float(np.median(aris)),loo_cluster_ari_min=float(min(aris)),cluster_mean_difference=diff,cluster_permutation_within_dataset_p=cp,largest_gain_sensitivity=dict(removed=rows[max_i]['name'],remaining=stats(y[trim]),groups={str(c):stats(y[trim&(labels==c)]) for c in [0,1] if np.any(trim&(labels==c))}),prediction_geometry=predict(rows,geom),prediction_geometry_texture=predict(rows,geom+additional[:2]),prediction_difficulty=predict(rows,['control_surface_psnr']))
order=np.argsort([x['p'] for x in associations]);q=np.minimum.accumulate((np.array([associations[i]['p'] for i in order])*len(order)/np.arange(1,len(order)+1))[::-1])[::-1]
for i,v in zip(order,q):associations[i]['bh_q']=float(min(1,v))
a={x['uid']:x for x in cohorts['heat_features']};b={x['uid']:x for x in cohorts['conditional_correction']};shared=sorted(set(a)&set(b));rho,p=spearman([a[u]['delta'] for u in shared],[b[u]['delta'] for u in shared])
result=dict(status='completed_post_hoc_cpu_analysis',plan=plan,cohorts=results,associations=associations,multiplicity_family=len(associations),cross_component=dict(n=len(shared),spearman=rho,permutation_p=p,exploratory_not_in_bh_family=True),scope='Existing outcomes and geometry/training-input descriptors; no new training, causal inference, or validated deployment policy.')
OUTPUT.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
for k,v in results.items():print(k,json.dumps({key:v[key] for key in ['overall','groups','silhouette','loo_cluster_ari_median','loo_cluster_ari_min','cluster_permutation_within_dataset_p','largest_gain_sensitivity']},ensure_ascii=False));print('LOO',k,{x:{j:v[x][j] for j in ['mae','baseline_mae','rmse','baseline_rmse','sign_accuracy']} for x in ['prediction_geometry','prediction_geometry_texture','prediction_difficulty']})
print('ASSOCIATIONS',json.dumps(sorted(associations,key=lambda x:x['p']),ensure_ascii=False));print('CROSS',result['cross_component'])
