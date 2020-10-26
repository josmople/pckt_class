import dataset as D
import torch
import pcapkit


from dataset import PCAPDataset


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


# x = pcapkit.extract(fin="D:/Datasets/ISCXVPN2016/hangouts_audio1a.pcap", format="pcap", layer="Internet")

ds = generate_pcap_dataset("D:/Datasets/ISCXVPN2016/hangouts_audio1a.pcap", size=40)
print(ds[0])
