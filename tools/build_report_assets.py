"""Build the heat-confirmation table/plot and core route plot from released records.

Other report figures are archived, provenance-labelled assets. This command does
not recreate omitted reconstruction images or the conceptual AI illustration.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from summarize_latest import summarize
from summarize_records import summarize as summarize_routes


def tex(text):
    return str(text).replace('&', r'\&').replace('_', r'\_').replace('#', r'\#').replace('%', r'\%')


def build(root, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42})
    output.mkdir(parents=True, exist_ok=True)
    data = summarize(root)
    heat = data['heat_features']
    aliases = {'Pokémon_Yellow_Special_Pikachu_Edition_Nintendo_Game_Boy_Color': 'Game Boy Color',
               'Olive_Kids_Butterfly_Garden_Pencil_Case': 'Butterfly pencil case',
               'Cascadia_8_Color_AquariusHibscsBearingSeaBlk_Size_50': 'Cascadia shoe',
               'Razer_Kraken_71_Chroma_headset_Full_size_Black': 'Kraken headset',
               'Razer_Naga_MMO_Gaming_Mouse': 'Naga mouse',
               'Mad_Gab_Refresh_Card_Game': 'Mad Gab game',
               'Twinlab_100_Whey_Protein_Fuel_Cookies_and_Cream': 'Whey protein tub',
               'DPC_tropical_Trends_Hat': 'Tropical hat',
               'علبة منتج': 'Product box (8 vertices)',
               '#SelfieTennis Ball Design Kit': 'SelfieTennis ball',
               'Mann holder kvinne i fanget': 'Seated figures',
               'Seahorse for 3d seascene': 'Seahorse'}
    rows = heat['primary_objects'] + heat['excluded_objects']
    lines = [r'\begin{table}[t]\centering\small',
             r'\caption{All 22 declared feature-confirmation objects. Gains are feature minus control in rendered PSNR. Two-seed means enter the primary result; incomplete or unsupported objects do not. Names are shortened; released records retain full names and IDs.}',
             r'\begin{tabular}{p{.48\linewidth}lrr}\toprule',
             r'Object & Source & Pairs & $\Delta$dB\\\midrule']
    for row in rows:
        name = aliases.get(row['name'], row['name'])
        delta = f"{row['mean_seed_delta']['pbr_psnr']:+.3f}" if row['qualified'] else ('single' if row['seeds'] else '--')
        lines.append(f"{tex(name)} & {'Obj.' if row['dataset']=='Objaverse' else 'GSO'} & {len(row['seeds'])} & {delta}" + r'\\')
    lines += [r'\bottomrule\end{tabular}\label{tab:heatobjects}\end{table}', '']
    (output/'heat_objects.tex').write_text('\n'.join(lines))
    primary = heat['primary_objects']
    order = sorted(primary, key=lambda x:x['mean_seed_delta']['pbr_psnr'])
    fig, ax = plt.subplots(figsize=(6.7, 4.9), layout='constrained')
    values = [o['mean_seed_delta']['pbr_psnr'] for o in order]
    ax.barh(range(len(order)), values, color=['#237ca7' if x>=0 else '#bd6857' for x in values])
    ax.set_yticks(range(len(order)), [aliases.get(o['name'], o['name']) for o in order], fontsize=8)
    ax.axvline(0, color='black', lw=.7); ax.set_xlabel('Feature minus control: rendered PSNR (dB)')
    ax.spines[['top','right']].set_visible(False)
    fig.savefig(output/'heat_confirmation.pdf'); plt.close(fig)
    evidence=json.loads((root/'results/evidence.json').read_text())
    routes=summarize_routes(evidence, 'mean_view_psnr')
    fig, ax=plt.subplots(figsize=(3.35,2.22),layout='constrained')
    refs=['HKTex','capped{b}','capped{b}_restart']
    found=[]
    for bi,b in enumerate([4000,3000]):
        selected=[]
        for ref in refs:
            match=[r for r in routes['seed0_summary'] if r['cohort']=='new_confirmation' and r['budget']==b and r['reference']==ref.format(b=b)]
            if len(match)!=1:raise ValueError('Missing unique fitting-route aggregate')
            selected.append(match[0]); found.append(match[0])
        ax.bar(np.arange(3)+(bi-.5)*.34,[r['mean'] for r in selected],width=.32,label=f'{b:,} kernels',color=['#237ca7','#e3a537'][bi])
    ax.axhline(0,color='black',lw=.7);ax.set_xticks(range(3),['Parent','Direct','Extended\ndirect'])
    ax.set_ylabel('Compress–refine gain (dB)');ax.legend(frameon=False,fontsize=7,loc='upper right')
    ax.spines[['top','right']].set_visible(False);fig.savefig(output/'fitting_question.pdf');plt.close(fig)
    (output/'computed_route_values.json').write_text(json.dumps(found,indent=2)+'\n')
    (output/'display_names.json').write_text(json.dumps({r['uid']:dict(original=r['name'],display=aliases.get(r['name'],r['name'])) for r in rows},indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(state='passed', generated=['heat_objects.tex','heat_confirmation.pdf','fitting_question.pdf','computed_route_values.json','display_names.json'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path,default=Path('outputs/report_assets'))
    args=parser.parse_args();build(args.root,args.output)
