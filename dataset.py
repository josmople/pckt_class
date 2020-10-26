import typing as _T
import pytorch_dataset as D

import torch


class PCAPDataset(D.model.ValueDataset):

    def __init__(self, path: str, transform: _T.Callable, layers: int = 0, verbose: bool = False, lazy: bool = False):
        from pcapfile.savefile import load_savefile

        self.file_raw = open(path, 'rb')
        self.file_parsed = load_savefile(self.file_raw, layers=layers, verbose=verbose, lazy=lazy)

        super().__init__(self.file_parsed.packets, transform)

    def close(self):
        self.file_raw.close()


def generate_pcap_dataset(path: str, size: int = 40, layers: int = 0, verbose: bool = False, lazy: bool = False):
    def transform(packet):
        bin_raw = packet.raw()[:size]
        bin_str = str.join("", map(lambda n: f"{n:b}", bin_raw))
        bin_arr = [float(c) for c in bin_str]
        return torch.tensor(bin_arr)

    ds = PCAPDataset(path=path, transform=transform, layers=layers, lazy=lazy, verbose=verbose)
    return ds
