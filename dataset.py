import typing as _T

import pytorch_dataset as utils

import torch


class PCAPDataset(utils.model.ValueDataset):

    def __init__(self, path: str, transform: _T.Callable, layers: int = 0, verbose: bool = False, lazy: bool = False):
        from pcapfile.savefile import load_savefile

        self.file_raw = open(path, 'rb')
        self.file_parsed = load_savefile(self.file_raw, layers=layers, verbose=verbose, lazy=lazy)

        super().__init__(self.file_parsed.packets, transform)

    def close(self):
        self.file_raw.close()


def generate_pcap_dataset(path: str, size: int = 40, verbose: bool = False, lazy: bool = False):
    from pcapfile.linklayer import ethernet

    def transform(packet):
        raw_bytes = ethernet.strip_ethernet(packet.raw())
        raw_size = len(raw_bytes)
        missing_size = max(0, size - raw_size)
        missing_bytes = bytes([0] * missing_size)
        all_data = raw_bytes + missing_bytes
        data = all_data[:size]
        data = data[:12] + data[20:]
        bitstring = str.join("", map(lambda n: f"{n:08b}", data))
        bitarray = [float(c) for c in bitstring]
        return torch.tensor(bitarray)

    ds = PCAPDataset(path=path, transform=transform, layers=0, lazy=lazy, verbose=verbose)
    return ds


class ISCXVPN2016Dataset(utils.model.ValueDataset):

    def __init__(self, path, transform=generate_pcap_dataset, include_filename=False):
        self.path = path
        self.raw_transform = transform
        self.include_filename = include_filename

        def value_transform(filename):
            filepath = f"{self.path}/{filename}"
            if self.include_filename:
                return filename, self.raw_transform(filepath)
            return self.raw_transform(filepath)

        super().__init__(self.load(path), value_transform)

    def load(self, path):
        from glob import glob
        from os.path import basename
        files = glob(f"{path}/*")
        files = map(basename, files)
        files = filter(lambda f: f.endswith(".pcap") or f.endswith(".pcapng"), files)
        return list(files)

    def reload(self):
        self.values = self.load(self.path)

    def find(self, filename):
        return self[self.values.index(filename)]

    def select(self, fn):
        return list(map(self.find, filter(fn, self.values)))


def generate_dummy_dataset(path, **kwds):
    from functools import partial
    transform = partial(generate_pcap_dataset, **kwds)
    ds = ISCXVPN2016Dataset(path, transform, include_filename=False)

    items = [
        ("facebook_audio1a.pcap", "facebook_audio2a.pcap"),
        ("skype_audio1a.pcap", "skype_audio2a.pcap"),
        ("email1a.pcap", "email1b.pcap", "email2a.pcap", "email2b.pcap"),
        ("spotify1.pcap", "spotify2.pcap"),
        ("vimeo1.pcap", "vimeo2.pcap"),
        ("hangouts_audio1a.pcap", "hangouts_audio2a.pcap")
    ]

    datasets = []
    for item in items:
        dataset = [ds.find(addr) for addr in item]
        dataset = sum(dataset[1:], dataset[0])
        datasets += [dataset]

    labels = []
    for i, dataset in enumerate(datasets):
        label = [i] * len(dataset)
        labels += [label]

    all_datasets = sum(datasets[1:], datasets[0])
    all_labels = sum(labels[1:], labels[0])

    return len(items), utils.dzip(all_datasets, all_labels)
    # facebook = ds.find("facebook_audio1a.pcap") + ds.find("facebook_audio2a.pcap")
    # skype = ds.find("skype_audio1a.pcap") + ds.find("skype_audio2a.pcap")
    # spotify = ds.find("spotify1.pcap") + ds.find("spotify2.pcap")
    # vimeo = ds.find("vimeo1.pcap") + ds.find("vimeo2.pcap")
    # hangout = ds.find("hangouts_audio1a.pcap") + ds.find("hangouts_audio2a.pcap")

    # facebook_labels = [0] * len(facebook)
    # skype_labels = [1] * len(skype)
    # spotify_labels = [2] * len(spotify)
    # vimeo_labels = [3] * len(vimeo)
    # hangout_labels = [4] * len(hangout)

    # packets = facebook + skype + spotify + vimeo + hangout
    # labels = facebook_labels + skype_labels + spotify_labels + vimeo_labels + hangout_labels

    # return utils.dzip(packets, labels)
