# coding: utf-8

lines = []
with open("pku_train") as f:
    for line in f:
        lines.append(''.join(line.split()))
        
get_ipython().magic('cd nn/')
with open("pku_train") as f:
    for line in f:
        lines.append(''.join(line.split()))
        
get_ipython().magic('cd ../working_data/')
with open("pku_train") as f:
    for line in f:
        lines.append(''.join(line.split()))
        
open("pku_test.raw") as f:
    for line in f:
        lines.append(''.join(line.split()))
        
with open("pku_test.raw") as f:
    for line in f:
        lines.append(''.join(line.split()))
        
len(lines)
lines[0]
lines[0][0]
w = {}
for line in lines:
    line = line.strip()
    if line:
        for c in line:
            count = w.get(c, 0)
            count += 1
            w[c] = count
            
len(w)
w
lw = [(k, v) for k, v in w.items()]
lw
get_ipython().magic('pinfo sorted')
get_ipython().magic('pinfo lw.sort')
lw = sorted(lw, key=lambda e: e[1])
lw[:10]
lw = sorted(lw, key=lambda e: e[1], reverse=True)
lw[:10]
lw[:100]
len(w)
