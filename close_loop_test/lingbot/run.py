"""Run one LingBot-Navi release-v1 static episode using 3DGS RGB and raw OCC execution."""
import argparse,json,os,signal,sys,time
from pathlib import Path
from tempfile import TemporaryDirectory
for key,value in [('OMP_NUM_THREADS','4'),('MKL_NUM_THREADS','4'),('OPENBLAS_NUM_THREADS','1')]:os.environ.setdefault(key,value)
from close_loop_test.common.observations import jsonable,save_observation
from close_loop_test.common.client import ModelClient
from .release import ReleaseV1

def run(args):
    source=ReleaseV1(args.dataset_release,args.manifest_sha256)
    if args.inspect:
        episodes={p.stem:len(source.content(p)['episodes']) for p in source.task_files()}
        print(json.dumps(dict(dataset=str(source.root),manifest_sha256=source.digest,protocol=source.atlas['protocol'],episodes=episodes),indent=2));return
    output=args.output.resolve()
    if output.is_relative_to(source.root):raise ValueError('Output must be outside the dataset release')
    output.mkdir(parents=True,exist_ok=False)
    result=dict(status='running',dataset='lingbot',dataset_release=str(source.root),manifest_sha256=source.digest,
        judge_protocol=source.atlas['protocol'],renderer='data_toolkit_native_pinhole_3dgs',depth_source='source_mesh_axial',
        navigation_executor='raw_occ_supercover_v1',uses_navmesh_execution=False,memory_geometry='lingbot_map1' if args.lingbot_map1 else 'simulator',arguments=jsonable(vars(args)))
    runtime=TemporaryDirectory(prefix='.runtime-', dir=output)
    work=Path(runtime.name)
    env=client=session=None;started=time.monotonic()
    def event(kind,**values):
        events.write(json.dumps(jsonable(dict(event=kind,elapsed_seconds=time.monotonic()-started,**values)),allow_nan=False)+'\n');events.flush()
    with (output/'events.jsonl').open('w') as events:
        try:
            from .environment import Environment
            options=json.loads(args.renderer_config.read_text()) if args.renderer_config else {}
            env=Environment(source,args.scene,args.episode_index,args.renderer_python,gpu=args.gpu,seed=args.seed,renderer_options=options)
            ep=env.current_episode
            result.update(episode_id=ep.episode_id,scene_id=ep.scene_id,tasks=ep.tasks,task_instructions=ep.task_instructions,start_adjustment=ep.lingbo['start_adjustment'])
            if args.environment_only:
                # Exercise actual motion and collision; diagnostic outcomes are never model scores.
                for action in ['turn_left','turn_right','look_down','look_up','move_forward']:
                    env.step(dict(action=action));event('environment_action',action=action,position=env.position(),collision=env.sim.occupancy_last_block)
                save_observation(env,env.sim._prev_sim_obs,work)
                result['status']='environment_checked'
            else:
                map1=None
                if args.lingbot_map1:
                    map1=dict(python=str(args.map1_python or args.model_python),checkpoint=str(args.map1_checkpoint.resolve()))
                client=ModelClient(args.model_python,output,dict(weights=str(args.weights.resolve()),base_model=str(args.base_model.resolve()),device=args.device,output=str(work),max_voxels=args.max_voxels,scene=ep.scene_id,map1=map1))
                result['model']=client.identity
                from .session import Session
                session=Session(env,client,work,args.budget,event,save_observation,map1=args.lingbot_map1,init_explore_steps=args.init_explore_steps,independent_initial_exploration=True)
                while session._episode_active and not session.done:session.episode_step()
                result.update(status='complete',metrics=env.get_metrics(),steps=session.step,plans=session.plans)
        except BaseException as error:
            result.update(status='failed',error=repr(error))
            if env is not None:result['partial_metrics']=env.get_metrics()
            raise
        finally:
            result['elapsed_seconds']=time.monotonic()-started
            (output/'result.json').write_text(json.dumps(jsonable(result),indent=2,allow_nan=False)+'\n')
            try:
                if client is not None:client.close()
                if env is not None:env.close()
            finally:
                runtime.cleanup()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-release',type=Path,required=True);p.add_argument('--manifest-sha256')
    p.add_argument('--inspect',action='store_true');p.add_argument('--scene');p.add_argument('--episode-index',type=int,default=0)
    p.add_argument('--output',type=Path);p.add_argument('--environment-only',action='store_true')
    p.add_argument('--renderer-python',type=Path);p.add_argument('--renderer-config',type=Path)
    for name in ('model-python','weights','base-model','map1-python','map1-checkpoint'):p.add_argument('--'+name,type=Path)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--gpu',type=int,default=0);p.add_argument('--device',default='cuda:0')
    p.add_argument('--budget',type=int,default=500);p.add_argument('--init-explore-steps',type=int,default=1000)
    p.add_argument('--max-voxels',type=int,default=500000)
    p.add_argument('--lingbot-map1',action=argparse.BooleanOptionalAction,default=False)
    a=p.parse_args()
    if a.base_model is None and a.weights is not None:
        weights = a.weights.expanduser().resolve()
        a.base_model = weights if weights.is_dir() else weights.parent
    if any(getattr(a,n)<=0 for n in ('budget','init_explore_steps','max_voxels')):p.error('budgets and max-voxels must be positive')
    if not a.inspect:
        if any(getattr(a,n) is None for n in ('scene','output','renderer_python')):p.error('scene, output and renderer-python are required')
        if not a.environment_only and any(getattr(a,n) is None for n in ('model_python','weights')):p.error('model-python and weights are required')
        if a.lingbot_map1 and (a.map1_checkpoint is None or a.environment_only):p.error('lingbot-map1.0 requires a checkpoint and a model evaluation')
    def interrupted(signum,frame):raise KeyboardInterrupt('Evaluation interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    run(a)

if __name__=='__main__':main()
