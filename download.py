import json
import os
import sys

filename=str(sys.argv[1])

filepath=os.path.join('/workspace/examples/mxnet/',filename)

content=[]
with open(filepath) as f:
    for line in f:
        content.append(json.loads(line))

os.chdir('/workspace/data/test')
name_dict={}
index=1
for i in content:
    url=i['url']
    suffix=url.rsplit('.',1)[1]
    if suffix =='gif':
        continue
    name=str(index)+'.'+suffix
    os.system('wget -O '+name+' '+url)
    name_dict.update({name:url})
    index=index+1

with open('/workspace/examples/mxnet/match_name.json','w') as op:
    json.dump(name_dict,op)