import json

with open('submission.json', 'r') as f:
   data = json.load(f)
   anns = data['annotations']
with open('submissionCF.json','w') as f:
   json.dump(anns,f)