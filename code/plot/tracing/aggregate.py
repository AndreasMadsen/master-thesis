
import pandas as pd


class Aggregate:
    def __init__(self, tracing):
        self.device_tree = DeviceMap()
        for device, ops in tracing.ops.items():
            self.device_tree.add_item(device, self._build_tree(device, ops))

    def _build_tree(self, device, ops):
        root = Node(device.name.upper().replace(':', ' - '))

        for op in ops:
            root.child('backward' if op.gradient else 'forward') \
                .child(op.major_group) \
                .child(op.minor_group) \
                .child(op.patch_group) \
                .child(op.name) \
                .add_event(op)

        return root

    def print(self):
        for tree in self.device_tree:
            print(f'{tree.name}: {tree.duration}')
            for ward in tree:
                print(f'| {ward.name}: {ward.duration}')
                for major in ward:
                    print(f'| | {major.name}: {major.duration}')
                    for minor in major:
                        print(f'| | | {minor.name}: {minor.duration}')

    def dataframe(self):
        data = []

        def append_data(tree, level, start, node):
            data.append({
                'device': tree.name,
                'level': level,
                'name': node.name,
                'start': start,
                'duration': node.duration
            })

        def accumulate_duration(offset, iterator):
            start = offset
            for node in iterator:
                yield start, node
                start += node.duration

        for tree in self.device_tree:
            for ward_start, ward in accumulate_duration(0, tree):
                append_data(tree, 0, ward_start, ward)

                for major_start, major in accumulate_duration(ward_start, ward):
                    append_data(tree, 1, major_start, major)

                    for minor_start, minor in accumulate_duration(major_start, major):
                        append_data(tree, 2, minor_start, minor)

        return pd.DataFrame(data, columns=(
                'device', 'level', 'name', 'start', 'duration'
            )
        )


class DeviceMap:
    def __init__(self):
        self.device_map = dict()

    def add_item(self, device, tree):
        self.device_map[device] = tree

    def __iter__(self):
        for device, tree in sorted(
            self.device_map.items(), key=lambda item: item[0].name
        ):
            yield tree


class Node:
    def __init__(self, name, parent=None):
        self.children = dict()
        self.parent = parent
        self.duration = 0
        self.name = name

    def add_event(self, event):
        self.duration += event.dur

        if self.parent is not None:
            self.parent.add_event(event)

    def child(self, name):
        if name not in self.children:
            self.children[name] = Node(name, parent=self)

        return self.children[name]

    def __iter__(self):
        return iter(
            sorted(self.children.values(), key=lambda val: val.name)
        )
