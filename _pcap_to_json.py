import pcapkit

input_file = "data/facebook_audio1a.pcap"
output_file = "out.json"

json = pcapkit.extract(fin=input_file, fout=output_file, format='json', extension=False)
