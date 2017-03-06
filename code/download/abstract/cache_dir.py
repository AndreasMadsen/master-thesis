
import os
import os.path as path

from code.download.util.download_dir import download_dir
from code.download.util.tqdm_download import download


class CacheDir:
    dirname: str
    dirpath: str

    def __init__(self, name: str) -> None:
        self.dirname = name
        self.dirpath = path.join(download_dir(), self.dirname)

    def __enter__(self):
        try:
            os.mkdir(self.dirpath, mode=0o755)
        except FileExistsError:
            pass

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def download(self, name: str, url: str) -> None:
        if not self.exists(name):
            print(f'downloading: {self.dirname}/{name} from {url}')
            download(url, self.filepath(name), desc=name)

    def exists(self, name: str) -> bool:
        return path.exists(self.filepath(name))

    def filepath(self, name: str) -> str:
        return path.join(self.dirpath, name)
