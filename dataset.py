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

    hangout = ds.find("hangouts_audio1a.pcap") + ds.find("hangouts_audio2a.pcap")
    facebook = ds.find("facebook_audio1a.pcap") + ds.find("facebook_audio2a.pcap")
    skype = ds.find("skype_audio1a.pcap") + ds.find("skype_audio2a.pcap")
    spotify = ds.find("spotify1.pcap") + ds.find("spotify2.pcap")
    vimeo = ds.find("vimeo1.pcap") + ds.find("vimeo2.pcap")

    hangout_labels = [0] * len(hangout)
    facebook_labels = [1] * len(facebook)
    skype_labels = [2] * len(skype)
    spotify_labels = [3] * len(spotify)
    vimeo_labels = [4] * len(vimeo)

    packets = hangout + facebook + skype + spotify + vimeo
    labels = hangout_labels + facebook_labels + skype_labels + spotify_labels + vimeo_labels

    return utils.dzip(packets, labels)
