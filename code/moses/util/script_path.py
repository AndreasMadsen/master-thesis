
import os.path as path


def script_path(filepath):
    this_dir = path.dirname(path.realpath(__file__))
    return path.realpath(
        path.join(this_dir, '..', '..', 'deps', 'mosesdecoder', filepath)
    )
