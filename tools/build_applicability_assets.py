"""Plot recorded continuation costs and post-hoc geometry associations."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'outputs/applicability_figures');args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
curves=json.loads((ROOT/'results/retained_studies/continuation_cost.json').read_text())
labels={'joint':'Prune + joint','prefit':'Colour fit + joint','alternating':'Periodic colour fitting','direct_joint':'Direct + joint'}
fig,axs=plt.subplots(2,2,figsize=(10,6),layout='constrained')
for ax,obj in zip(axs.flat,curves):
 for arm,label in labels.items():
  pts=obj['arms'][arm]['points'];ax.plot([p['seconds'] for p in pts],[p['psnr'] for p in pts],'.-',label=label,lw=1.3,ms=4)
 ax.axvline(obj['common_budget']['seconds'],color='.5',ls=':',lw=1);ax.set(title=obj['name'].replace('_',' '),xlabel='Accounted pipeline time (s)',ylabel='Rendered PSNR (dB)');ax.grid(alpha=.15)
axs[0,0].legend(fontsize=8,loc='lower right');fig.savefig(args.output/'quality_cost_curves.pdf');plt.close(fig)
x=json.loads((ROOT/'results/applicability/results.json').read_text());fig,axs=plt.subplots(1,2,figsize=(10,3.4),layout='constrained')
for ax,(key,title) in zip(axs,[('heat_features','Fixed-time heat features'),('conditional_correction','Conditional correction')]):
 rows=x['cohorts'][key]['objects']
 for cluster in [0,1]:
  for dataset,marker in [('objaverse','o'),('gso','s')]:
   chosen=[r for r in rows if r['geometry_cluster']==cluster and r['dataset'].lower()==dataset]
   if not chosen:continue
   ax.scatter([10**r['features']['log10_vertices'] for r in chosen],[r['delta'] for r in chosen],c=['#D58F35','#357CAD'][cluster],marker=marker,s=36,edgecolors='white',linewidths=.5)
 ax.set_xscale('log');ax.set(title=title,xlabel='Mesh vertices',ylabel='Rendered PSNR change (dB)');ax.axhline(0,color='.4',lw=.8);ax.grid(alpha=.15)
 # Label the actual largest positive effect and the counterexample, without deleting any data.
 for needle in (['Doormate','Football'] if key=='heat_features' else ['Football','Doormate']):
  rr=next(r for r in rows if needle in r['name']);ax.annotate(needle,(10**rr['features']['log10_vertices'],rr['delta']),xytext=(7,5),textcoords='offset points',fontsize=8)
from matplotlib.lines import Line2D
handles=[Line2D([],[],marker='o',ls='',color=c,label=n) for c,n in [('#D58F35','Geometry cluster 1'),('#357CAD','Geometry cluster 2')]]
handles+=[Line2D([],[],marker=m,ls='',color='.5',label=n) for m,n in [('o','Objaverse'),('s','GSO')]]
fig.legend(handles=handles,ncol=4,loc='outside lower center',frameon=False,fontsize=8)
fig.savefig(args.output/'applicability.pdf');plt.close(fig)
print('Wrote two measured-data figures; no training or interpolation.')
