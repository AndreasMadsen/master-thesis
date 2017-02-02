
import os
import os.path as path
import urllib.request

from code.dataset.util.download_dir import download_dir


class WMTEnv:
    wmt_dir: str

    def __init__(self, wmt_dir: str=None) -> None:
        if wmt_dir is None:
            wmt_dir = path.join(download_dir(), 'wmt')

        self.wmt_dir = wmt_dir

    def __enter__(self):
        # create wmt dir
        try:
            os.mkdir(self.wmt_dir, mode=0o755)
        except FileExistsError:
            pass

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def download(self, name: str, url: str) -> None:
        if not path.exists(self.filepath(name)):
            urllib.request.urlretrieve(url, self.filepath(name))
            print(f'downloading: wmt/{name} from {url}')

    def filepath(self, name: str) -> str:
        return path.join(self.wmt_dir, name)
