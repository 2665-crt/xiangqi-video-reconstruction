import json
import os
import sys
import time
from pathlib import Path

experiment = Path(__file__).resolve().parents[1]
application = experiment
sys.path.insert(0,str(experiment/'backend'))
sys.path.append(str(application/'.venv/Lib/site-packages'))
os.environ['VISION_FUSION']='1'
import cv2
import numpy as np
from app.vision import VisionClient

out=experiment/'output/fusion-validation'
out.mkdir(parents=True,exist_ok=True)
source=application/'output/github-recognition-study'
manifest=json.loads((source/'evaluation-manifest.json').read_text(encoding='utf8'))
baseline=json.loads((source/'baseline-result.json').read_text(encoding='utf8'))
vision=VisionClient()
boards=[]
for board in manifest['boards']:
    image=cv2.imdecode(np.fromfile(board['rectified_path'],np.uint8),cv2.IMREAD_COLOR)
    started=time.perf_counter()
    candidates=vision.recognize_position([image],actual=True)
    diagnostic=vision.last_response
    draft=diagnostic['draft_position']['pieces']
    predicted={(p['square']['file'],p['square']['rank']):(p['color'],p['kind']) for p in draft}
    gold={(p['file'],p['rank']):(p['color'],p['kind']) for p in board['pieces']}
    assert all(abs(sum(r['class_probabilities'].values())-1)<1e-5 for r in diagnostic['readings'] if r['class_probabilities'])
    row={'name':board['name'],'group':board['group'],'gold_count':len(gold),
         'correct':sum(predicted.get(s)==v for s,v in gold.items()),'predicted_count':len(predicted),
         'candidate_available':bool(candidates),'seconds':time.perf_counter()-started,'diagnostic':diagnostic}
    boards.append(row)
    print(json.dumps({k:v for k,v in row.items() if k!='diagnostic'},ensure_ascii=False),flush=True)
groups={}
for group in sorted({b['group'] for b in boards}):
    subset=[b for b in boards if b['group']==group]
    count=sum(b['gold_count'] for b in subset)
    correct=sum(b['correct'] for b in subset)
    groups[group]={'count':count,'correct':correct,'accuracy':correct/count,
                   'baseline':baseline['summary'][group]['overall']}
report={'groups':groups,'boards':boards,'probability_sums_verified':True,
        'evaluation_limit':'Reused development samples, not new blind evaluation; wood 3/4 and held video contain partial labels.'}
(out/'real-frame-result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'groups':groups},ensure_ascii=False),flush=True)
vision.close()
