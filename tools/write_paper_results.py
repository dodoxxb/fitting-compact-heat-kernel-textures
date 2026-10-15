"""Generate all expanded tables/plots from the failure-aware evidence ledger."""
from pathlib import Path
from collections import defaultdict
import json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--evidence',required=True,type=Path)
parser.add_argument('--diagnostic',required=True,type=Path)
parser.add_argument('--output',required=True,type=Path)
parser.add_argument('--allow-preview',action='store_true')
args=parser.parse_args()
P=args.output.resolve();P.mkdir(parents=True,exist_ok=True);(P/'figures').mkdir(exist_ok=True)
D=json.loads(args.evidence.read_text())
if D['state']!='terminal_audited' and not args.allow_preview:
    parser.error('Final tables require terminal_audited evidence; use --allow-preview for a visibly marked draft.')

LABELS={'new_confirmation':'New Objaverse','localhk_main':'Public main','localhk_lowpoly':'Public low-poly','gso_external':'GSO','hktex_official_examples':'Tuning examples'}
GROUPS=list(LABELS)
def finite(v):return isinstance(v,(int,float)) and math.isfinite(v)
def tex(s):return str(s).replace('&',r'\&').replace('_',r'\_').replace('%',r'\%').replace('#',r'\#')
def write(name,lines):(P/name).write_text('\n'.join(lines)+'\n')
def stat(c,k,ref,metric='mean_view_psnr'):
    return next((s for s in D['summary'] if (s['cohort'],s['budget'],s['reference'],s['metric'])==(c,k,ref,metric)),None)
def cell(s):return '--' if s is None else f"{s['mean']:+.3f} ({s['objects']})"
def start(cols,small='small'):return [r'\begin{table}[ht]\centering'+chr(92)+small,r'\begin{tabular}{'+cols+r'}\toprule']
def end(caption,label):return [r'\bottomrule\end{tabular}',r'\caption{'+caption+r'}\label{'+label+r'}\end{table}']

write('build_status.tex',[] if D['state']=='terminal_audited' else [r'\begin{center}\textbf{Incomplete working draft: pending results are not submission evidence.}\end{center}'])
mac=[]
for k,word in [(4000,'Four'),(3000,'Three')]:
    for ref,label in [('HKTex','Parent'),(f'capped{k}','Direct'),(f'capped{k}_restart','Extended')]:
        s=stat('new_confirmation',k,ref);mac.append(r'\newcommand{\New'+word+label+'}{'+f"{s['mean']:+.3f}"+'}')
    s=stat('new_confirmation',k,'HKTex');mac.append(r'\newcommand{\New'+word+'Savings}{'+f"{-100*s['mean_storage_change']:.1f}"+'}')
    s=stat('new_confirmation',k,f'capped{k}_restart')
    for label,v in [('Wins',s['positive']),('N',s['objects'])]:mac.append(r'\newcommand{\New'+word+label+'}{'+str(v)+'}')
write('expanded_numbers.tex',mac)

lines=start('lrrrrr')+[r'Cohort & Planned & Valid ref. & Eligible 4k/3k & Valid refined 4k/3k\\\midrule']
# Five columns, with paired capacities sharing a cell.
lines[1]=r'\begin{tabular}{lrrrr}\toprule'
for c in GROUPS:
    cov=[r for r in D['coverage'] if r['cohort']==c]
    valid=sum(r['parent_valid'] for r in cov)
    elig=[sum(r['parent_valid'] and (r['parent_sources'] or 0)>k for r in cov) for k in [4000,3000]]
    fits=[sum(r['cohort']==c and r['seed']==0 and r['method']==f'warm{k}' for r in D['measurements']) for k in [4000,3000]]
    lines.append(f"{LABELS[c]} & {len(cov)} & {valid} & {elig[0]}/{elig[1]} & {fits[0]}/{fits[1]}"+r'\\')
lines+=end('Locked object coverage. A valid reference has a finite exported field; exact-match image views do not invalidate it. Eligibility requires the reference count to exceed the requested cap. Valid refined counts require completed decoded evaluation. The tuning examples are descriptive.','tab:coverage')
write('expanded_coverage.tex',lines)

lines=start('llrrrr','scriptsize')+[r'Cohort & Budget & Reference & Direct & Extended direct & Full continuation\\\midrule']
for c in GROUPS:
    for k in [4000,3000]:
        cells=[cell(stat(c,k,ref)) for ref in ['HKTex',f'capped{k}',f'capped{k}_restart','restart']]
        lines.append(LABELS[c]+f' & {k:,} & '+' & '.join(cells)+r'\\')
lines+=end('Mean compress--refine minus comparison PSNR in dB; parentheses give finite paired-object counts. All scores here use the declared arithmetic mean of per-view PSNR. Different columns may have different denominators after invalid or undefined endpoints. Positive values favour compress--refine.','tab:expanded-quality')
write('expanded_quality.tex',lines)

lines=start('llrr','small')+[r'Cohort & Budget & Edited reference & Rebuilt-support refit\\\midrule']
for c in GROUPS:
    for k in [4000,3000]:
        lines.append(LABELS[c]+f' & {k:,} & '+cell(stat(c,k,f'posthoc{k}'))+' & '+cell(stat(c,k,f'posthoc_rebuilt{k}'))+r'\\')
lines+=end('Quality recovered by compress--refine over the two post-hoc colour-fitting controls, in mean-view PSNR dB (finite paired counts). Post-hoc fitting distils the reference; refinement revisits source appearance and uses additional optimization, so these margins do not isolate a geometry effect.','tab:posthoc-expanded')
write('posthoc_expanded.tex',lines)

methods=defaultdict(dict)
for row in D['measurements']:
    if row['seed']==0:methods[row['uid']][row['method']]=row
lines=start('llrrrr','scriptsize')+[r'Cohort & Budget & Objects & Coded reference KiB & Coded refined KiB & PSNR $\Delta$\\\midrule']
codecsummary=[]
for c in GROUPS:
    for k in [4000,3000]:
        paired=[]
        for uid,m in methods.items():
            a=m.get('parent_C8B12S8');b=m.get(f'warm{k}_C8B12S8')
            if a and b and a['cohort']==c:paired.append((a,b))
        if paired:
            pb=float(np.median([a['bytes']/1024 for a,b in paired]));wb=float(np.median([b['bytes']/1024 for a,b in paired]))
            ds=[b['pooled_psnr']-a['pooled_psnr'] for a,b in paired if finite(a['pooled_psnr']) and finite(b['pooled_psnr'])]
            delta='--' if not ds else f'{np.mean(ds):+.3f} ({len(ds)})'
            cells=f'{len(paired)} & {pb:.2f} & {wb:.2f} & {delta}'
            codecsummary.append(dict(cohort=c,budget=k,objects=len(paired),median_coded_parent_kib=pb,median_coded_refined_kib=wb,mean_pooled_psnr_delta=float(np.mean(ds)) if ds else None))
        else:cells='0 & -- & -- & --'
        lines.append(LABELS[c]+f' & {k:,} & '+cells+r'\\')
lines+=end('The same scalar codec applied to paired reference and refined fields. Sizes are medians of actual bitstreams on the indicated common objects. Quality uses supplementary pooled PSNR; its finite count is shown in parentheses. This tests count reduction alongside coding, not superiority over every coded training route.','tab:codec-expanded')
write('codec_expanded.tex',lines)
(P/'codec_summary.json').write_text(json.dumps(codecsummary,indent=2))

fig,axs=plt.subplots(1,2,figsize=(10.8,3.6),sharey=True)
colors=['#22759b','#bd6942']; display=GROUPS[:4]
for ax,refname,title in zip(axs,['HKTex','extended'],'Compared with the reference|Compared with extended direct'.split('|')):
    for ci,c in enumerate(display):
        for ki,k in enumerate([4000,3000]):
            ref=refname if refname=='HKTex' else f'capped{k}_restart'
            rs=[r for r in D['pairs'] if r['cohort']==c and r['budget']==k and r['seed']==0 and r['reference']==ref and finite(r['mean_view_psnr'])]
            x=np.array([r['mean_view_psnr'] for r in rs]);y=np.full(len(x),ci+(ki-.5)*.24)
            if len(x):
                ax.scatter(x,y,s=13,alpha=.52,color=colors[ki],label=f'{k:,} kernels' if ci==0 else None)
                ax.scatter([np.mean(x)],[y[0]],s=62,marker='D',color=colors[ki],edgecolor='white',zorder=3)
    ax.axvline(0,color='#777',lw=.8);ax.set_title(title,fontsize=10);ax.set_xlabel('Paired PSNR change (dB)');ax.grid(axis='x',alpha=.15)
    ax.spines[['top','right']].set_visible(False)
axs[0].set_yticks(range(len(display)),[LABELS[c] for c in display]);axs[0].invert_yaxis();axs[1].legend(frameon=False,fontsize=9,loc='lower right')
fig.tight_layout();fig.savefig(P/'figures/expanded_effects.pdf',bbox_inches='tight');plt.close(fig)

# Deployment dominance uses MSE, so an exact-match UV is retained.
byuid=defaultdict(list)
for r in D['uv']:
    if finite(r.get('pooled_mse')):byuid[r['uid']].append(r)
lines=start('lrrrr','scriptsize')+[r'Cohort & Refined 4k & Refined 3k & Coded refined 4k & Coded refined 3k\\\midrule']
deployment=[]
for c in GROUPS:
    cells=[]
    for name in ['warm4000','warm3000','warm4000_C8B12S8','warm3000_C8B12S8']:
        rs=[r for r in D['measurements'] if r['cohort']==c and r['seed']==0 and r['method']==name and r['uid'] in byuid]
        wins=[]
        for r in rs:
            dom=any(u['bytes']<=r['bytes'] and u['pooled_mse']<=r['pooled_mse'] and (u['bytes']<r['bytes'] or u['pooled_mse']<r['pooled_mse']) for u in byuid[r['uid']])
            wins.append(dom)
        cells.append(f'{sum(wins)}/{len(rs)}' if rs else '--')
        deployment.append(dict(cohort=c,method=name,uv_dominates=sum(wins),compared=len(rs)))
    lines.append(LABELS[c]+' & '+' & '.join(cells)+r'\\')
lines+=end('Objects on which at least one tested native UV texture is no larger and has no higher pooled image MSE than the kernel field, with at least one strict improvement. Exact-match UV textures are included through zero MSE. Denominators require valid kernel and UV measurements. Counts describe the tested byte/quality candidates, not all codecs or runtime memory.','tab:uv-dominance')
write('deployment_expanded.tex',lines)
(P/'deployment_summary.json').write_text(json.dumps(deployment,indent=2))

r29=json.loads(args.diagnostic.read_text());assert r29['state']=='complete'
lines=[r'\paragraph{Numerical validity is a separate requirement.}',
    'One GSO booster-seat object produces nonfinite parameters under the default refinement despite finite logged losses at the larger budget. Inspection finds the same invalid parameters in the full-precision checkpoint, ruling out export-only corruption. In a selected-case replay, the first observed nonfinite orientation gradient occurs at update 374/630 for the 4,000/3,000-kernel fields. These observations localize the onset but do not isolate its ultimate mathematical cause.',
    'Reducing all learning rates by a factor of ten yields valid fields at both budgets; a hundredfold reduction is also finite but recovers less quality within the same updates (Table~\\ref{tab:numerical}). The original failed endpoints remain failures in the expanded comparison. This diagnostic does not establish a universal scale or a tested automatic recovery rule.']
lines+=start('lrr')+[r'Continuation scale & 4,000 kernels & 3,000 kernels\\\midrule']
for scale,label in [(1.,'Default'),(.1,'One tenth'),(.01,'One hundredth')]:
    cells=[]
    for k in [4000,3000]:
        row=next(r for r in r29['runs'] if r['budget']==k and r['lr_scale']==scale)
        cells.append(f"{row['metrics']['psnr']:.2f}" if row['state']=='complete' else 'Invalid')
    lines.append(label+' & '+' & '.join(cells)+r'\\')
lines+=end('Selected failure diagnosis, PBR PSNR in dB. Each continuation uses the same original initializer, seed and 5,000 updates. The default-scale replay reproduces invalidity. These six experiments are outside the locked confirmation.','tab:numerical')
write('numerical_validity.tex',lines)

lines=[r'\section{Expanded results and measurement accounting}\label{app:expanded}',
    'This appendix preserves the distinction between the declared image score and supplementary aggregation. Pooled PSNR is computed from the MSE over all five images before taking the logarithm. It is applied uniformly, including objects with some exact-match views. Bootstrap intervals are descriptive object resamples within each finite paired cohort; no sequential significance stopping is used.']
lines+=start('llrrrr','scriptsize')+[r'Cohort & Budget & View-mean $\Delta$ & Pooled $\Delta$ & LPIPS $\Delta$ & SSIM $\Delta$\\\midrule']
for c in GROUPS:
    for k in [4000,3000]:
        metrics=[]
        for m in ['mean_view_psnr','pooled_psnr','lpips','ssim']:
            s=stat(c,k,'HKTex',m);metrics.append('--' if s is None else f"{s['mean']:+.3f} ({s['objects']})" if m.endswith('psnr') else f"{s['mean']:+.5f}")
        lines.append(LABELS[c]+f' & {k:,} & '+' & '.join(metrics)+r'\\')
lines+=end('Metric sensitivity for compress--refine versus the reference. PSNR cells give finite paired counts; smaller LPIPS and larger SSIM are favourable. Neither a positive mean PSNR nor one aggregation convention implies universal or perceptual improvement.','tab:metrics')
lines+=start('llrr','scriptsize')+[r'Cohort & Budget & Reference: mean [95\% interval] & Extended direct: mean [95\% interval]\\\midrule']
for c in GROUPS:
    for k in [4000,3000]:
        cells=[]
        for ref in ['HKTex',f'capped{k}_restart']:
            s=stat(c,k,ref);cells.append('--' if not s else f"{s['mean']:+.3f} [{s['bootstrap95'][0]:+.3f}, {s['bootstrap95'][1]:+.3f}]")
        lines.append(LABELS[c]+f' & {k:,} & '+' & '.join(cells)+r'\\')
lines+=end('Descriptive bootstrap intervals for the mean-view PSNR difference, using 10,000 object resamples. These convenience cohorts do not justify population-wide confidence guarantees.','tab:intervals')
lines+=['The expanded Objaverse/public collection has eleven full spectral-cache failures (five SVD nonconvergence and six embedding-rank assertions) and one further reference-initialization failure caused by texture-array dimensionality. Ten of the cache failures and the initialization failure belong to the new Objaverse panel; the remaining cache failure is an official tuning example. On GSO, the missing smaller-budget Spider-Man fit fails CUDA initialization before training begins; it is an environment failure, not evidence of an optimization breakdown. The booster-seat case instead has a nonfinite larger-budget export and a failed smaller-budget optimization. No selected-case repair changes these confirmation endpoints. The machine-readable ledger distinguishes these causes and records every planned training arm.']
outcomes=D.get('training_outcomes',[])
if outcomes:
    lines+=start('lrrrrrr','scriptsize')+[r'Route & Valid & Env. & Fit fail & Invalid & Ineligible & Blocked\\\midrule']
    route_names={'parent':'Reference','capped4000':'Direct 4k','capped3000':'Direct 3k','warm4000':'Refined 4k','warm3000':'Refined 3k','restart':'Full continuation','capped4000_restart':'Extended direct 4k','capped3000_restart':'Extended direct 3k'}
    for arm,label in route_names.items():
        rs=[r for r in outcomes if r['arm']==arm];states=[r['state'] for r in rs]
        counts=[states.count(s) for s in ['valid_evaluated','environment_failed','training_failed','invalid_export','budget_ineligible']]
        counts.append(sum(s.startswith('not_run_') for s in states))
        lines.append(label+' & '+' & '.join(map(str,counts))+r'\\')
    lines+=end('Training-arm accounting over all 91 prescribed object--seed slots. Valid means a finite field with completed evaluation, including exact-match image views. Blocked arms lack a required cache, reference or initializer; they are not additional independent failures. Fit failures include initialization failures. The original environment and numerical failures remain missing outcomes.','tab:arm-coverage')
lines+=start('lrrrr','scriptsize')+[r'Repeated object & Ref. 4k & Extended 4k & Ref. 3k & Extended 3k\\\midrule']
for obj in D['coverage']:
    if len(obj['expected_seeds'])!=3:continue
    cells=[]
    for k in [4000,3000]:
        for ref in ['HKTex',f'capped{k}_restart']:
            ps=[p for p in D['pairs'] if p['uid']==obj['uid'] and p['budget']==k and p['reference']==ref and finite(p['mean_view_psnr'])]
            cells.append(f"{np.mean([p['mean_view_psnr'] for p in ps]):+.3f} (3/3)" if len(ps)==3 else f'-- ({len(ps)}/3)')
    name=obj['name'] or obj['uid'][:8]
    if len(name)>25:name=name[:22]+'...'
    lines.append(tex(name)+' & '+' & '.join(cells)+r'\\')
lines+=end('Object means over the three prescribed seeds. A mean is reported only when all three paired differences are finite; parentheses show valid coverage. Missing or budget-ineligible cases are not replaced. Seed0 cohort averages and these object-level repeat means answer different questions.','tab:seeds-expanded')
write('expanded_details.tex',lines)
print(json.dumps(dict(state=D['state'],uv_rows=len(D['uv']),figure=str(P/'figures/expanded_effects.pdf'))))
