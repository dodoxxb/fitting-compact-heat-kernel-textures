"""Recorded checkpoints only; full endpoint costs remain in the ledger."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]

def main():
    report=json.loads((ROOT/'results/final_followup/REPLAY_RESULTS.json').read_text())
    assert report['state']=='complete'
    fig,axes=plt.subplots(2,2,figsize=(6.5,4.4))
    for ax,record in zip(axes.flat,report['pairs']):
        for key,label,color in [('fp64_diagonal','Verified diagonal','#4477aa'),('heat_features','Fixed-time features','#cc6677')]:
            curve=record['variants'][key]['curve']
            ax.plot([p['accounted_seconds'] for p in curve],[p['pbr_psnr'] for p in curve],'-o',ms=3,lw=1.2,color=color,label=label)
        name='Barricade' if record['pair']['uid'].startswith('6e509') else 'Wrench'
        ax.set_title(f"{name}, seed {record['pair']['seed']}",fontsize=10)
        ax.set_xlabel('Cache + targets + training (s)',fontsize=9)
        ax.set_ylabel('Rendered PSNR (dB)',fontsize=9)
        ax.tick_params(labelsize=8);ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False,fontsize=9)
    fig.tight_layout(rect=(0,0,1,.93))
    (ROOT/'outputs').mkdir(exist_ok=True)
    fig.savefig(ROOT/'outputs/heat_feature_cost.pdf',bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':main()
