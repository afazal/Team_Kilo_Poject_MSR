
#This script is used to Parse aggregated all-contributors file.

import json
import pickle
from datetime import datetime

with open('scratch_output.txt', 'r') as f:
    data = dict()
    for line in f:
        if line[0:6] == '#begin':
            accu = []
            sha = line.strip().split("\t")[1]
            time = datetime.strptime(line.strip().split("\t")[2], "%a %b %d %H:%M:%S %Y")
        elif line[0:4] == "#end":
            try:
                data[(sha, time)] = json.loads(" ".join(accu))
            except json.decoder.JSONDecodeError:
                data[(sha, time)] = None  #invalid json
        else:
            accu += [line.strip()]

with open('/scratch_pickled.pck', 'wb') as f:
    pickle.dump(data, f)
