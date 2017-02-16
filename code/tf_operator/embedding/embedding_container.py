
from typing import Dict, List, Iterator, Tuple
import os.path as path
import os

import tensorflow as tf


class EmbeddingContainer:
    _embedding_label_map: Dict[tf.Tensor, List[str]] = {}

    def __init__(self):
        pass

    def add(self, embedding: tf.Tensor, labels: List[str]) -> None:
        if (embedding.get_shape()[0] != len(labels)):
            raise ValueError(f'embedding have shape {embedding.get_shape()}'
                             ' but there are {len(labels)} labels')

        # if we are in reuse mode, a label map should allready have been added
        if tf.get_variable_scope().reuse:
            return

        if embedding in self._embedding_label_map:
            if self._embedding_label_map[embedding] != labels:
                raise ValueError(f'pairing labels to {embedding.name} but' +
                                 ' they differ from existing labels')
        else:
            self._embedding_label_map[embedding] = labels

    def _filename(self, embedding) -> str:
        return f'{hash(embedding.name)}.metadata.tsv'

    def save_metadata(self, save_dir: str) -> None:
        if not path.exists(save_dir):
            os.makedirs(save_dir)

        for embedding, labels in self._embedding_label_map.items():
            metadata = '\n'.join(labels)
            filename = self._filename(embedding)
            filepath = path.join(save_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as fd:
                print(metadata, file=fd)

    def __iter__(self) -> Iterator[Tuple[tf.Tensor, str]]:
        for embedding in self._embedding_label_map.keys():
            yield (embedding, self._filename(embedding))

    def __len__(self) -> int:
        return len(self._embedding_label_map)
