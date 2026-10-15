"""Recompute final deployment choices and controls from released raw records."""
from pathlib import Path
import argparse,json,statistics as st,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research/round44'))
from selection import choose_precision,choose_readout

def stats(xs):
 xs=list(xs)
 return dict(n=len(xs),mean=st.mean(xs),median=st.median(xs),min=min(xs),max=max(xs))
def ps(r,k):return r['render']['metrics'][k]['mean_per_view']['psnr']
def summarize(d):
 rows=[]
 assert len(d['policies'])==16 and len(d['details'])==len(d['effort_control'])==4
 for r in d['policies']:
  c=r['choices'];m=c['calibration_source_mse'];p=c['precision'];s=c['readout']
  assert choose_precision(m,r['candidate_container_bytes'])==p
  assert choose_readout(m)==s
  rows.append(dict(uid=r['uid'],arm=r['arm'],precision=p,readout=s,
   precision_pbr_delta=ps(r,p)-ps(r,'Ref8'),precision_surface_delta=r['test_source'][p]['psnr']-r['test_source']['Ref8']['psnr'],precision_byte_ratio=r['payload_bytes']['precision']/r['payload_bytes']['reference8'],
   readout_pbr_delta=ps(r,s)-ps(r,'Stream6'),readout_surface_delta=r['test_source'][s]['psnr']-r['test_source']['Stream6']['psnr'],readout_speedup=r['render']['timing']['Stream6']/r['render']['timing'][s],
   guard_test_pass=r['agreement']['test']['Guard6']['pass_2e_5'],guard_all_pass=all(v['Guard6']['pass_2e_5'] for v in r['agreement'].values()),stream_test_pass=r['agreement']['test']['Stream6']['pass_2e_5'],guard_stream_speedup=r['render']['timing']['Stream6']/r['render']['timing']['Guard6']))
 keys=['precision_pbr_delta','precision_surface_delta','precision_byte_ratio','readout_pbr_delta','readout_surface_delta','readout_speedup','guard_stream_speedup']
 objects=[dict(uid=u,**{k:st.mean(r[k] for r in rows if r['uid']==u) for k in keys}) for u in sorted({r['uid'] for r in rows})]
 assert len(objects)==8
 counts=dict(fields=len(rows),precision_bits={k:sum(r['precision']==k for r in rows) for k in ['Ref4','Ref6','Ref8']},all_response_selected=sum(r['readout']=='All6' for r in rows),guard_all_splits_pass=sum(r['guard_all_pass'] for r in rows),guard_test_pass=sum(r['guard_test_pass'] for r in rows),stream_test_pass=sum(r['stream_test_pass'] for r in rows))
 for family in ['precision','readout']:
  for endpoint in ['pbr','surface']:counts[family+'_'+endpoint+'_within_005']=sum(r[family+'_'+endpoint+'_delta']>=-.05 for r in rows)
 details=[];effort=[]
 for r in d['details']:
  m=r['methods'];budget=m['Soft64']['bytes'];choice=r['locked_alternative_choice'];eligible=[k for k in ['Base','Hard32','Hard64'] if m[k]['bytes']<=budget]
  assert eligible==choice['feasible'];a=min(eligible,key=lambda k:m[k]['validation_mse']);assert a==choice['alternative']==r['selected_alternative']
  details.append(dict(uid=r['uid'],name=r['name'],alternative=a,soft_base=ps(r,'Soft64')-ps(r,'Base'),soft_hard64=ps(r,'Soft64')-ps(r,'Hard64'),soft_feasible=ps(r,'Soft64')-ps(r,a),soft_base_byte_ratio=budget/m['Base']['bytes'],soft_base_time_ratio=r['render']['timing']['Soft64']/r['render']['timing']['Base'],soft_feasible_surface=m['Soft64']['test_source_psnr']-m[a]['test_source_psnr']))
 for r in d['effort_control']:
  m=r['methods'];effort.append(dict(uid=r['uid'],name=r['name'],soft_base600=ps(r,'Soft64')-ps(r,'Base600'),base600_base300=ps(r,'Base600')-ps(r,'Base300'),soft_base600_surface=m['Soft64']['test_source_psnr']-m['Base600']['test_source_psnr'],soft_base600_byte_ratio=m['Soft64']['bytes']/m['Base600']['bytes'],soft_base600_time_ratio=r['render']['timing']['Soft64']/r['render']['timing']['Base600'],base600_gap=r['base_continuation']['half_mse_convex_gap']))
 out=dict(state='verified',counts=counts,policy_fields=rows,policy_objects=objects,policy_object_summary={k:stats(r[k] for r in objects) for k in keys},details=details,detail_summary={k:stats(r[k] for r in details) for k in details[0] if k not in ['uid','name','alternative']},effort=effort,effort_summary={k:stats(r[k] for r in effort) for k in effort[0] if k not in ['uid','name']},scope='CPU record and frozen-choice verification; no new independent objects or GPU replay.')
 return out
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,default=ROOT/'results/final_policies.json');p.add_argument('--output',type=Path);a=p.parse_args();s=json.dumps(summarize(json.loads(a.input.read_text())),indent=2)+'\n'
 if a.output:a.output.write_text(s)
 print(s)
