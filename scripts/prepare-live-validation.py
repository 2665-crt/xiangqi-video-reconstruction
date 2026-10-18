import json
import sys
from fractions import Fraction
from pathlib import Path

import av
import httpx

experiment=Path(__file__).resolve().parents[1]
application=experiment
sys.path.insert(0,str(experiment/'backend'))
from app.video import iter_frames

record=json.loads((application/'output/piece-classifier-validation/user-live/independent-annotations.json').read_text(encoding='utf8'))['samples'][-1]
out=experiment/'output/fusion-validation'
out.mkdir(parents=True,exist_ok=True)
clip=out/'real-held-video-840s.mp4'
with av.open(str(clip),'w') as container:
    stream=None
    frames=iter_frames(record['source_video'],record['time_s'],record['time_s']+1)
    try:
        for index,(_,image) in enumerate(frames):
            if index>=24:
                break
            if stream is None:
                stream=container.add_stream('libx264',rate=12)
                stream.width,stream.height=image.shape[1],image.shape[0]
                stream.pix_fmt='yuv420p'
            frame=av.VideoFrame.from_ndarray(image,format='bgr24')
            frame.pts=index
            frame.time_base=Fraction(1,12)
            for packet in stream.encode(frame):
                container.mux(packet)
    finally:
        frames.close()
    for packet in stream.encode():
        container.mux(packet)
with httpx.Client(base_url='http://127.0.0.1:8278',trust_env=False,timeout=30) as client:
    with clip.open('rb') as stream:
        response=client.post('/api/tasks',files={'file':('融合实验：真实视频840秒片段.mp4',stream,'video/mp4')})
    response.raise_for_status()
    task=response.json()
    response=client.patch(f'/api/tasks/{task["id"]}/setup',json={
        'revision':task['revision'],'start_s':0,'end_s':task['metadata']['duration_s'],
        'calibration':record['calibration']})
    response.raise_for_status()
    task=response.json()
(out/'live-task.json').write_text(json.dumps({'task_id':task['id'],'clip':str(clip),'calibration':record['calibration'],
    'source_video_sha256':record['source_video_sha256'],'source_start_s':record['time_s'],
    'purpose':'Actual video frames re-encoded for local API/UI pipeline validation, not full-game accuracy.'},
    ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'task_id':task['id'],'duration':task['metadata']['duration_s'],'calibration_confirmed':task['calibration_confirmed']}))
