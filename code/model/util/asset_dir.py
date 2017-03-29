import os
import os.path as path


def asset_dir() -> str:
    if 'ASSET_DIR' in os.environ:
        return path.realpath(os.environ['ASSET_DIR'])
    else:
        this_dir = path.dirname(path.realpath(__file__))
        return path.realpath(path.join(this_dir, '..', '..', 'asset'))
