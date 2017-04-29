
import json
import re


class Group:
    def __init__(self, tracing):
        self.ops = self._split_ops(tracing.devices, tracing.events)

    def _split_ops(self, devices, events):
        op_by_device = {}
        for event in events:
            if ModelOp.is_model_op(event):
                op = ModelOp(event)
                device = devices[op.pid]
                if device not in op_by_device:
                    op_by_device[device] = []
                else:
                    op_by_device[device].append(op)

        return op_by_device


class ModelOp:
    def __init__(self, event):
        self.pid = event['pid']
        self.name = event['name']
        self.dur = event['dur'] / 1e6
        self.ts = event['ts'] / 1e6

        modelpath = event['args']['name'].split('/')

        if modelpath[0] == 'train':
            self.gradient = True
            modelpath = modelpath[2:]
        else:
            self.gradient = False

        if modelpath[1] in ['encoder', 'decoder']:
            self.major_group = modelpath[1]

            for i in (3, 4):
                if modelpath[i] in ['conv-dilated', 'activation', 'recover-dim', 'reduce-dim']:
                    self.minor_group = modelpath[i]

                    if modelpath[i + 1] in ['batchnorm', 'moments']:
                        self.patch_group = 'normalization'
                    elif modelpath[i + 1] == 'conv1d':
                        self.patch_group = 'Dense'
                    elif modelpath[i + 1] == 'convolution':
                        self.patch_group = 'convolution'
                    else:
                        self.patch_group = 'other'
                    break
            else:
                self.minor_group = 'other'
                self.patch_group = modelpath[4]
        elif 'embedding' in modelpath[1]:
            self.major_group = 'other'
            self.minor_group = 'embedding'
            self.patch_group = modelpath[1]
        else:
            self.major_group = 'other'
            self.minor_group = 'other'
            self.patch_group = modelpath[1]

    @staticmethod
    def is_model_op(event):
        fast_check = event['ph'] == 'X' and event['cat'] == 'Op'
        if not fast_check:
            return False

        match = re.search(
            '^(train/gradients(_[0-9])?/)?bytenet-model(_[0-9])?/',
            event['args']['name']
        )
        return match is not None
