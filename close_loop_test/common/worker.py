"""Release model process: online memory and navigation protocol requests."""
import contextlib
import json
import sys
import traceback

def serve():
    model=memory=planner=None
    import signal
    def stop(signum, frame): raise SystemExit(0)
    signal.signal(signal.SIGTERM, stop)
    try:
        for line in sys.stdin:
            job=json.loads(line)
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    op=job['operation']
                    if op=='initialize':
                        import torch
                        torch.set_num_threads(4)
                        from lingbot_navi import LingBotNavi
                        from .memory import OnlineMemory
                        from .planning import Planner
                        model=LingBotNavi.from_pretrained(job['weights'],device=job['device'])
                        memory=OnlineMemory(job['base_model'],job['device'],job['output'],model.metadata,job.get('max_voxels',500000),job['scene'],map1=job.get('map1'))
                        planner=Planner(model,memory)
                        result=dict(protocol=model.metadata['joint_protocol']['version'],goal_mode='viewpoint' if model.metadata['joint_protocol']['version'].endswith('_v4') else 'v3_object_center')
                    elif op=='observe':result=memory.observe(job['bundle'],job.get('frame_id'))
                    elif op=='finish':result=memory.finish(job['segment'],job['phase'])
                    elif op=='localize':result=planner.localize(job)
                    elif op=='plan':result=planner.plan(job)
                    else:raise ValueError('Unknown operation: '+op)
                response=dict(id=job['id'],ok=True,result=result)
            except Exception as error:
                traceback.print_exc(file=sys.stderr);response=dict(id=job.get('id'),ok=False,error=str(error))
            print(json.dumps(response,allow_nan=False),flush=True)
    finally:
        if memory is not None: memory.close()

if __name__=='__main__':serve()
