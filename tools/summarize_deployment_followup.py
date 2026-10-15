"""Recompute deployment-pilot headline arithmetic from included raw records."""
from pathlib import Path
import json,statistics as st,argparse

ROOT=Path(__file__).resolve().parents[1]
def stats(xs):
 xs=list(xs)
 return dict(n=len(xs),mean=st.mean(xs),median=st.median(xs),min=min(xs),max=max(xs)) if xs else None
def ps(r,key):return r['render']['metrics'][key]['mean_per_view']['psnr']
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,default=ROOT/'results/deployment_followup.json');p.add_argument('--output',type=Path);a=p.parse_args()
 d=json.loads(a.input.read_text());r=d['rounds'];out={}
 assert all(len(r[str(n)])==(4 if n==39 else 16) for n in range(36,43))
 both=[x for x in r['37'] if x['agreement']['Stream']['pass_2e_5'] and x['agreement']['Compiled']['pass_2e_5']]
 out['streaming']=dict(qualified_fields=len(both),sampled_stream_passes=sum(x['agreement']['Stream']['pass_2e_5'] for x in r['37']),render_speedup=stats(x['render']['timing']['Compiled']/x['render']['timing']['Stream'] for x in both),temporary_ratio=stats(x['timing_8192_queries']['512']['Stream']['memory']['incremental_peak']/x['timing_8192_queries']['512']['Compiled']['memory']['incremental_peak'] for x in both),scope='Component compilation enabled; control additionally compiles whole queries. Temporary memory is not total VRAM.')
 solvers=[]
 for x in r['38']:
  v={m:dict(ok=all(t['converged'] for t in x['results'] if t['mode']==m),seconds=st.median(t['solve_seconds'] for t in x['results'] if t['mode']==m)) for m in ['global_accel','diagonal_plain','diagonal_accel']}
  solvers.append(v)
 out['solver']=dict(passes={m:sum(s[m]['ok'] for s in solvers) for m in solvers[0]},diagonal_seconds=stats(s['diagonal_accel']['seconds'] for s in solvers),qualified_speedup=stats(s['global_accel']['seconds']/s['diagonal_accel']['seconds'] for s in solvers if s['global_accel']['ok'] and s['diagonal_accel']['ok']))
 # Equal route counts make the field mean and paired-object mean identical;
 # compute the object aggregates explicitly to retain the sampling unit.
 uids=sorted({x['uid'] for x in r['40']});assert len(uids)==8
 six=[dict(uid=u,gain=st.mean(ps(x,'Fitted6')-ps(x,'Nearest6') for x in r['40'] if x['uid']==u),vs8=st.mean(ps(x,'Fitted6')-ps(x,'Nearest8') for x in r['40'] if x['uid']==u),ratio=st.mean(x['methods']['Fitted6']['bytes']/x['methods']['Nearest8']['bytes'] for x in r['40'] if x['uid']==u)) for u in uids]
 out['six_bit']=dict(objects=six,gain=stats(x['gain'] for x in six),vs8=stats(x['vs8'] for x in six),byte_ratio=stats(x['ratio'] for x in six))
 out['backend_gate']=dict(raw_passes=sum(x['query_pass_2e_5'] for x in r['41']),policy_passes=sum(x['policy_heldout_pass'] for x in d['policies']),total=len(d['policies']),failed=[dict(uid=x['uid'],arm=x['arm'],error=x['policy_heldout_max']) for x in d['policies'] if not x['policy_heldout_pass']])
 out['all_responses']=dict(fields=len(r['42']),speedup=stats(x['render']['timing']['Inner30']/x['render']['timing']['All50Fitted'] for x in r['42']),psnr_change=stats(ps(x,'All50Fitted')-ps(x,'Inner30') for x in r['42']),scope='Field statistics; two correlated routes per object; two discovery and six fixed-expansion objects.')
 out['scope']='CPU summary reproduction, not a new GPU run or fresh training reproduction.'
 encoded=json.dumps(out,indent=2)+'\n'
 if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(encoded)
 print(encoded)
if __name__=='__main__':main()
