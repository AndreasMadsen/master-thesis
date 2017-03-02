
import os
import os.path as path


def download_dir() -> str:
    this_dir = path.dirname(path.realpath(__file__))
    download_dir = path.realpath(
        path.join(this_dir, '..', '..', '..', 'download')
    )

    # create download dir
    try:
        os.mkdir(download_dir, mode=0o755)
    except FileExistsError:
        pass

    return download_dir
