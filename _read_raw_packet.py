
from pcapfile.savefile import load_savefile

with open("data/facebook_audio1a.pcap", 'rb') as f:
    p = load_savefile(f, verbose=True)

print(p.packets[1].raw()[:40])
