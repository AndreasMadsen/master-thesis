
import math
from typing import Optional

import sugartensor as stf

from model.abstract.model import Model
from dataset.abstract.text_dataset import TextDataset


class PrintDataset(Model):
    limit: Optional[int]

    def __init__(self, dataset: TextDataset, limit=None) -> None:
        self.limit = dataset.num_observation
        if limit is not None:
            self.limit = min(self.limit, limit)

        super().__init__(dataset)

    def train(self) -> None:
        # index formatter
        digits = math.ceil(math.log10(self.limit))
        formatter = '{:>%d}' % (digits, )
        source_formatter = '  ' + formatter + ' source: '
        target_formatter = '  ' + (' ' * digits) + ' target: '

        # evaluate tensor list with queue runner
        with stf.Session() as sess:
            stf.sg_init(sess)
            with stf.sg_queue_context():
                observations_read = 0

                for minibatch in range(self.dataset.num_batch):
                    print('minibatch: %d' % minibatch)

                    sources, targets = sess.run([
                        self.dataset.source, self.dataset.target
                    ])

                    for i, source, target in zip(
                        range(observations_read, self.limit),
                        sources,
                        targets
                    ):
                        observations_read = i + 1

                        print(source_formatter.format(i) +
                              self.dataset.decode_as_str(source))
                        print(target_formatter +
                              self.dataset.decode_as_str(target))

                    print('')

                    if observations_read == self.limit:
                        break
