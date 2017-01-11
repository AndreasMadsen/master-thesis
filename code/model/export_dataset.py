
import math
from typing import Optional

import sugartensor as stf

from model.abstract.model import Model
from dataset.abstract.text_dataset import TextDataset


class ExportDataset(Model):
    limit: Optional[int]

    def __init__(self, dataset: TextDataset, limit=None) -> None:
        self.limit = dataset.num_observation
        if limit is not None:
            self.limit = min(self.limit, limit)

        self.targets = []
        self.sources = []

        super().__init__(dataset)

    def train(self) -> None:
        # evaluate tensor list with queue runner
        with stf.Session() as sess:
            stf.sg_init(sess)
            with stf.sg_queue_context():
                observations_read = 0

                for minibatch in range(self.dataset.num_batch):

                    sources, targets = sess.run([
                        self.dataset.source, self.dataset.target
                    ])

                    for i, source, target in zip(
                        range(observations_read, self.limit),
                        sources,
                        targets
                    ):
                        observations_read = i + 1

                        self.sources.append(self.dataset.decode_as_str(source))
                        self.targets.append(self.dataset.decode_as_str(target))

                    if observations_read == self.limit:
                        break
