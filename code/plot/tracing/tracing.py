
import json
import re

import pandas as pd


class Tracing:
    def __init__(self, filename):
        with open(filename) as fd:
            data = json.load(fd)
            self.events = data['traceEvents']

        self.devices = self._merge_devices()

    def _merge_devices(self):
        device_by_name = {}
        device_by_pid = {}

        for event in self.events:
            if Device.is_device(event):
                device = Device(event)

                if device.name in device_by_name:
                    device_by_name[device.name].add_device(device)
                    device_by_pid[device.pid] = device_by_name[device.name]
                else:
                    device_by_name[device.name] = device
                    device_by_pid[device.pid] = device

        return device_by_pid

    def dataframe(self):
        data = []

        for event in self.events:
            if event['ph'] == 'X' and event['cat'] == 'Op':
                device = self.devices[event['pid']]
                data.append({
                    'device': device.name.upper().replace(':', ' - '),
                    'name': event['name'],
                    'thread': event['tid'],
                    'start': event['ts'] / 1e6,
                    'duration': event['dur'] / 1e6
                })

        dataframe = pd.DataFrame(data, columns=(
                'device', 'name', 'thread', 'start', 'duration'
            )
        )
        dataframe['start'] = dataframe['start'] - dataframe['start'].min()
        return dataframe


class Device:
    def __init__(self, event):
        self.pid = event['pid']
        self.pids = [self.pid]

        if event['args']['name'] == 'Allocators':
            self.type = 'Allocators'
            self.index = 0
            self.name = 'Allocators'
        else:
            match = re.search(
                '^/job:localhost/replica:0/task:0/(cpu|gpu):([0-9]+) ',
                event['args']['name']
            )
            self.type = match.group(1)
            self.index = int(match.group(2))
            self.name = f'{self.type}:{self.index}'

    def add_device(self, device):
        self.pids += device.pids

    @staticmethod
    def is_device(event):
        return event['ph'] == 'M' and event['name'] == 'process_name'
