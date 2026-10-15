"""Evaluate the frozen scalar codec and native UV ladder for a trained object."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher', type=Path, required=True)
    parser.add_argument('--queries', type=Path, required=True)
    parser.add_argument('--models', required=True, help='Comma-separated name:NPZ pairs, e.g. parent:outputs/run/parent/model.npz,warm:outputs/run/warm/model.npz')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--codec-modes', default='C8B12S8',
                        help='Comma-separated fixed codec modes; use C8B12S8,C8B14S10,C8B16S12 for the three-precision control.')
    parser.add_argument('--skip-uv', action='store_true',
                        help='Evaluate only the requested coded models, without repeating the UV ladder.')
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a new output directory.')
    if ',' in args.gpu:
        parser.error('Select exactly one GPU.')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(ROOT/'research/round21'))
    from joint_codec import encode,decode,modes
    import numpy as np
    selected_modes=args.codec_modes.split(',')
    available=modes()
    if not selected_modes or len(set(selected_modes))!=len(selected_modes) or any(m not in available for m in selected_modes):
        parser.error('Choose unique codec modes from: '+','.join(available))
    report=dict(state='running',codec={},excluded={},codec_modes=selected_modes,
                uv_requested=not args.skip_uv)
    model_specs=[]
    seen_names=set()
    for item in args.models.split(','):
        if ':' not in item:
            parser.error('Each model must use name:path syntax.')
        name,path=item.split(':',1)
        if not name.replace('_','').isalnum():
            parser.error('Use alphanumeric model labels.')
        if name in seen_names:
            parser.error('Repeated model label: '+name)
        seen_names.add(name)
        model_specs.append((name,path))
    models=[]
    for name,path in model_specs:
        path=Path(path).resolve()
        with np.load(path,allow_pickle=False) as model:
            valid=all(np.isfinite(v).all() for v in model.values())
        if not valid:
            report['excluded'][name]='nonfinite_model';continue
        for mode in selected_modes:
            label=name+'_'+mode
            encoded=out/(label+'.jhk');decoded=out/(label+'_decoded.npz')
            report['codec'][label]=encode(path,encoded,available[mode])
            np.savez_compressed(decoded,**decode(encoded))
            models.append((label,decoded))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,CUDA_DEVICE_ORDER='PCI_BUS_ID',
             TORCH_COMPILE_DISABLE='1',POINT_BATCHING='512',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    env['PYTHONPATH']=str(ROOT)+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    def run(label,script,arguments):
        with (out/(label+'.log')).open('w') as log:
            subprocess.run([sys.executable,str(ROOT/script),*map(str,arguments)],
                           cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    (out/'controls.json').write_text(json.dumps(report,indent=2))
    try:
        if models:
            run('codec','research/round22/evaluate_variants.py',[
                '--teacher',args.teacher.resolve(),'--queries',args.queries.resolve(),
                '--output',out/'coded_evaluation','--models',','.join(f'{k}:{v}' for k,v in models)])
        if not args.skip_uv:
            run('uv','research/round28/uv_reference.py',['--teacher',args.teacher.resolve(),
                '--queries',args.queries.resolve(),'--output',out/'uv'])
        report['state']='complete'
    except Exception as exc:
        report.update(state='failed',error=repr(exc));raise
    finally:
        (out/'controls.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
