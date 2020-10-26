import typing as _T
import pytorch_dataset as D


class PCAPDataset(D.model.ValueDataset):

    def __init__(self, path: str, transform: _T.Callable, layers: int = 0, verbose: bool = False, lazy: bool = False):
        from pcapfile.savefile import load_savefile

        self.file_raw = open(path, 'rb')
        self.file_parsed = load_savefile(self.file, layers=layers, verbose=verbose, lazy=lazy)

        super().__init__(self.file_parsed.packets, transform)

    def close(self):
        self.file_raw.close()


def generate_pcap_dataset(path: str, size: int = 40, layers: int = 0, verbose: bool = False, lazy: bool = False):
    def transform(packet):
        return packet.raw()[:size]

    ds = PCAPDataset(path=path, transform=transform, layers=layers, lazy=lazy, verbose=verbose)
    return ds
