"""Export native USD assets after Kit loads PhysX schemas; no physics stepping."""
import argparse
from pathlib import Path
import importlib.util
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device='cpu', multi_gpu=False).app
    source = Path(__file__).parent
    spec = importlib.util.spec_from_file_location('_backup_service', source/'task.py',
                                                submodule_search_locations=[str(source)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    try:
        for name, path in module.create_assets(args.output).items(): print(name, path)
    finally:
        app.close()
