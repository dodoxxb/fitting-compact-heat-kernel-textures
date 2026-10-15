"""Build the audited DiGeo initial-edge correction; no GPU allocation required."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path)
    args = parser.parse_args()
    import torch
    from torch.utils.cpp_extension import load
    base = ROOT / 'research/round15/geometry_diagnostic/boundary_check'
    build = (args.build_dir or base / 'build').resolve()
    build.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'native/geodesic_boundary_diag.cu'
    library = load(name='geodesic_boundary_diag', sources=[str(source)],
                   build_directory=str(build), extra_cuda_cflags=['-lineinfo'],
                   is_python_module=False, verbose=True)
    base.mkdir(parents=True, exist_ok=True)
    (base / 'build.json').write_text(json.dumps({
        'library': library, 'torch': torch.__version__,
        'patched_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'scope': 'Initial local edge index correction and isolated operator namespace; BSD-3-Clause.'
    }, indent=2))
    print(library)


if __name__ == '__main__':
    main()
