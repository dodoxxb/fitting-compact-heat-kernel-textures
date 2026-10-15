"""Recompute scalar summaries and figures from shipped mechanism records."""
from pathlib import Path
import argparse,json
import numpy as np

def summarize(r):
 out=dict(scope=r['scope'],render_conditions=[],sampling={},weighted={},heat=[])
 for res in [128,256,512]:
  for alpha in [.125,.25,.5,1.]:
   v=[x['delta_mean_view_psnr'] for o in r['renders'] for x in o['rows'] if x['resolution']==res and x['alpha']==alpha]
   lin=[x['raw_linear_delta_pooled_psnr'] for o in r['renders'] for x in o['rows'] if x['resolution']==res and x['alpha']==alpha]
   out['render_conditions'].append(dict(resolution=res,alpha=alpha,mean=float(np.mean(v)),positive=sum(x>0 for x in v),n=len(v),min=min(v),max=max(v),linear_pooled_mean=float(np.mean(lin))))
 for m in ['original16k','fresh16k','larger65k']:
  a=[x for x in r['sampling'] if x['method']==m];out['sampling'][m]=dict(n=len(a),**{k:float(np.mean([x[k] for x in a])) for k in ['test_delta_db','direction_energy_test_over_fit','projected_gram_difference_spectral_norm','active_box_fraction','half_mse_gap']})
 for label,source in [('100sweeps',r['weighted_sampling']),('gap_stopped',r.get('converged',[]))]:
  a=[x for x in source if x['method']=='uniform'];b=[x for x in source if x['method']=='projected_leverage']
  if not a:continue
  delta=[next(y for y in b if y['name']==x['name'] and y['seed']==x['seed'])['selected_test_delta_db']-x['selected_test_delta_db'] for x in a]
  out['weighted'][label]=dict(n_pairs=len(delta),selected_mean_difference=float(np.mean(delta)),positive_pairs=sum(x>0 for x in delta),methods={})
  for m in ['uniform','projected_leverage']:
   v=[x for x in source if x['method']==m];out['weighted'][label]['methods'][m]={k:float(np.mean([x[k] for x in v])) for k in ['full_test_delta_db','selected_test_delta_db','half_mse_gap']}
 if 'converged' in r:
  out['converged_pools']={m:dict(n=4,full_test_delta_db=float(np.mean([x['full_test_delta_db'] for x in r['converged'] if x['method']==m]))) for m in ['original16k','fresh16k','larger65k']}
  out['converged_qualified']=sum(x['qualified'] for x in r['converged']);out['converged_count']=len(r['converged']);out['converged_max_gap']=max(x['half_mse_gap'] for x in r['converged'])
 for o in r['heat']['objects']:
  out['heat'].append(dict(name=o['name'],n=len(o['records']),response_better=sum(x['response_relative_error']<x['factor_relative_error'] for x in o['records']),**{k:float(np.mean([x[k] for x in o['records']])) for k in ['factor_relative_error','response_relative_error','factor_to_response_trace','factor_sign_change_relative_difference']}))
 return out

def assets(r,s,out):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'ps.fonttype':42})
 fig,axs=plt.subplots(1,3,figsize=(7,2.55),sharey=True)
 names={'PoisonDartFrog':'Frog','PinkOmbreHair':'Hair','SmallLimo':'Limo','Orange':'Orange'}
 for ax,res in zip(axs,[128,256,512]):
  for o in r['renders']:
   row=sorted([x for x in o['rows'] if x['resolution']==res],key=lambda x:x['alpha']);ax.plot([x['alpha'] for x in row],[x['delta_mean_view_psnr'] for x in row],'-o',ms=3,label=names[o['name']])
  ax.axhline(0,color='.5',lw=.6);ax.set_title(f'{res} × {res} actual render');ax.set_xlabel('Colour update fraction');ax.grid(alpha=.15);ax.set_xticks([0,.25,.5,1])
 axs[0].set_ylabel('Mean-view PSNR change (dB)');axs[-1].legend(fontsize=6,loc='lower left',frameon=False);fig.tight_layout();fig.savefig(out/'mechanism_render_path.pdf');plt.close(fig)
 rows=[r'\begin{table*}[t]',r'\centering\small',r'\begin{tabular}{lrrrrrr}',r'\toprule',r'& \multicolumn{2}{c}{128$^2$} & \multicolumn{2}{c}{256$^2$} & \multicolumn{2}{c}{512$^2$} \\',r'Object & Partial & Full & Partial & Full & Partial & Full \\ \midrule']
 for o in r['renders']:
  v=[next(x['delta_mean_view_psnr'] for x in o['rows'] if x['resolution']==res and x['alpha']==a) for res in [128,256,512] for a in [.25,1.]];rows.append(names[o['name']]+' & '+' & '.join(f'{x:+.3f}' for x in v)+r' \\')
 rows += [r'\bottomrule\end{tabular}',r'\caption{Actual colour-path renders: change in mean-view PSNR (dB) relative to the same-resolution initial field. Partial means $\alpha=1/4$, selected on surface validation for all four objects before rendering; full means $\alpha=1$. All three resolutions use five original cameras. These are four reused development objects, not twelve independent objects.}\label{tab:render-path}',r'\end{table*}'];(out/'mechanism_render_table.tex').write_text('\n'.join(rows)+'\n')
 rows=[r'\begin{table}[t]\centering\small',r'\begin{tabular}{lrrr}\toprule',r'Fit samples & Test $\Delta$ & Energy ratio & Active (\%) \\ \midrule']
 for m,label in [('original16k','Original 16k'),('fresh16k','Fresh 16k'),('larger65k','Nested 65k')]:
  v=s['sampling'][m];rows.append(f"{label} & {v['test_delta_db']:+.3f} & {v['direction_energy_test_over_fit']:.3f} & {100*v['active_box_fraction']:.1f}"+r' \\')
 rows += [r'\bottomrule\end{tabular}',r'\caption{Twelve correlated events on four objects, 100-sweep full colour updates. Test $\Delta$ is surface PSNR in dB. Energy ratio is independent-test over fitting mean squared field change. Active is the fraction of non-background colour entries at a box bound.}\label{tab:sampling-mechanism}',r'\end{table}'];(out/'mechanism_sampling_table.tex').write_text('\n'.join(rows)+'\n')
 rows=[r'\begin{table}[t]\centering\small',r'\begin{tabular}{lrrr}\toprule',r'Mesh & Factor error & Response error & Trace ratio \\ \midrule']
 for x in s['heat']:rows.append(f"{x['name']} & {x['factor_relative_error']:.3f} & {x['response_relative_error']:.3f} & {x['factor_to_response_trace']:.3f}"+r' \\')
 rows += [r'\bottomrule\end{tabular}',r'\caption{Same nodal heat factors, nine off-grid parameter points per mesh. Error is relative Frobenius discrepancy from a fresh same-rank off-grid eigensolution on 128 fixed vertices. Trace ratio is factor over response interpolation. These operator metrics are not rendered texture quality.}\label{tab:interpolation-mechanism}',r'\end{table}'];(out/'mechanism_heat_table.tex').write_text('\n'.join(rows)+'\n')
 if 'gap_stopped' in s['weighted']:
  rows=[r'\begin{table}[t]\centering\small',r'\begin{tabular}{llrr}\toprule',r'Solve & Sampling & Full & Selected \\ \midrule']
  for lab,title in [('100sweeps','100 sweeps'),('gap_stopped','Gap stopped')]:
   for m,label in [('uniform','Uniform'),('projected_leverage','Projected leverage')]:
    x=s['weighted'][lab]['methods'][m];rows.append(f"{title} & {label} & {x['full_test_delta_db']:+.3f} & {x['selected_test_delta_db']:+.3f}"+r' \\')
  rows += [r'\bottomrule\end{tabular}',r'\caption{Surface-test PSNR change (dB), four objects and three sampling seeds. Both replacement samplers estimate the same finite-pool quadratic risk. Selected steps use only surface validation; projection, weighting and selection costs are additional.}\label{tab:weighted-mechanism}',r'\end{table}'];(out/'mechanism_weighted_table.tex').write_text('\n'.join(rows)+'\n')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--assets',action='store_true');a=p.parse_args();r=json.loads(a.input.read_text());s=summarize(r);a.output.mkdir(parents=True,exist_ok=True);(a.output/'summary.json').write_text(json.dumps(s,indent=2,allow_nan=False)+'\n')
 if a.assets:assets(r,s,a.output)
 print(json.dumps(s,indent=2))
