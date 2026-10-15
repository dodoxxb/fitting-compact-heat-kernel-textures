"""Make the evaluator checkpoint consumed by the original calibration scripts.

This serializes the full-precision state and resolved configuration only. The
representation evaluated later is model.npz, independently decoded by the
unchanged evaluator. Historical export-time diagnostic draws are unnecessary.
"""
import argparse
import dataclasses
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher', required=True, type=Path)
    args = parser.parse_args()
    import torch
    from omegaconf import DictConfig, ListConfig, OmegaConf

    def plain(value):
        if isinstance(value, (DictConfig, ListConfig)):
            return OmegaConf.to_container(value, resolve=True)
        if dataclasses.is_dataclass(value):
            return plain(dataclasses.asdict(value))
        if isinstance(value, dict):
            return {k: plain(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [plain(v) for v in value]
        return value

    # This is the user's own just-trained checkpoint, not a downloaded pickle.
    state = torch.load(args.teacher / 'state.pt', map_location='cpu', weights_only=False)
    config = plain(state['trainer_config'])
    model_config = config['model']
    model_config['n_sources'] = len(state['model']['_kernel_colours'])
    mesh = json.loads((args.teacher / 'status.json').read_text())['mesh']
    torch.save(dict(model=state['model'], model_config=model_config,
                    eigen_config=config['eigen_albo'], mesh_path=mesh,
                    loader_kwargs=dict(merge_tex=True, normalise_size=True)),
               args.teacher / 'official_smoke_checkpoint.pt')


if __name__ == '__main__':
    main()
