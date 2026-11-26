"""Shared retrieval/frontier protocol using the release public model API."""
import numpy as np
from scipy.spatial.transform import Rotation
from lingbot_navi.navigation_transformer.prompts import IMAGE_ABSENT, frontier_messages
from .policy.routes import rank_retrieval_centers
from .policy.task_memory_selection import rigid_frame, select_coverage, select_frame_anchor, memory_frame_counts

def paired_viewpoints(values, centers):
    if not len(centers) and (values is None or len(values) == 0):
        return np.empty((0, 5))
    poses = np.asarray(values, dtype=np.float64)
    if poses.shape != (len(centers), 5) or not np.isfinite(poses).all():
        raise ValueError('grounding requires one finite XYZ/cos/sin viewpoint per center')
    norms = np.linalg.norm(poses[:, 3:], axis=1)
    if np.any(norms < 1e-8):
        raise ValueError('viewpoint has undefined heading')
    poses = poses.copy()
    poses[:, 3:] /= norms[:, None]
    return poses

class Planner:
    def __init__(self, model, memory):
        self.model,self.memory=model,memory

    def localize(self, job):
        paths=job['memories']
        prepared,frames,xyz=self.memory.prepare(paths)
        out=self.predict(prepared,'Localize',image=job['image'],task='localization',return_block_diagnostics=job.get('select',False))
        pose,answer=self.pose(out,allow_absent=True)
        frame = rigid_frame(frames[0])
        reference = np.eye(4)
        reference[:3,:3] = frame[:3,:3].T
        reference[:3,3] = -frame[:3,:3].T @ frame[:3,3]
        raw=dict(pose=None if pose is None else pose.tolist(),answer=answer,
                 reference_block_from_odom=reference.tolist())
        if job.get('select'):
            predictions=out['block_predictions']
            selection=select_coverage(paths,xyz,predictions,job['newest_first'],horizontal_rotation=np.asarray(frames[0]['rotation_matrix']))
            selection=select_frame_anchor(selection,memory_frame_counts(paths))
            _,selected_frames,_=self.memory.prepare(selection['selected'])
            raw.update(selection=selection,block_predictions=predictions,reference_block_frames=frames,selected_anchor_frame=selected_frames[0])
        return raw

    @staticmethod
    def pose(out,allow_absent=False):
        answer=out['generated_text'].replace('<|im_end|>','').strip()
        poses=np.asarray(out.get('predicted_poses') or [])
        if out.get('predicted_centers') or out.get('trajectory'):raise ValueError('localization returned center/trajectory')
        if not poses.size and allow_absent and answer==IMAGE_ABSENT:return None,answer
        if poses.shape!=(1,7) or not np.isfinite(poses).all() or not answer:raise ValueError('Invalid localization: '+answer)
        return poses[0],answer

    def predict(self,*args,**kwargs):
        out=self.model.predict(*args,**kwargs)
        # v19 coordinate navigation ends at the trajectory queries, without EOS.
        allowed_finish_reasons = {'eos'}
        if kwargs.get('task') == 'coordinate':
            allowed_finish_reasons.add('trajectory_complete')
        if out.get('finish_reason') not in allowed_finish_reasons or out.get('coordinate_frame')!='first_block_meters':
            raise ValueError('Invalid model termination/frame: '+str(out.get('finish_reason')))
        return out

    def plan(self,job):
        try:
            return self._plan(job)
        except Exception as error:
            # Source _dialogue_mode failure falls through to semantic exploration.
            if job['target']['kind']=='image' or job.get('force_explore') or job.get('exploration_goal_world') is not None:
                raise
            result=self._plan(dict(job,force_explore=True))
            result['retrieval_error']=str(error)
            return result

    def _plan(self,job):
        prepared,frames,xyz=self.memory.prepare(job['memories'])
        current=job['current_localization']; pose=np.asarray(current['pose'])
        world=np.linalg.inv(np.asarray(current['block_from_odom']))
        rotation,origin=world[:3,:3],world[:3,3]
        target=job['target'];image=target['kind']=='image'
        mode='image' if image else 'dialogue'
        retrieval=None;goal=None;extra={}
        if job.get('exploration_goal_world') is not None:
            goal=rotation.T@(np.asarray(job['exploration_goal_world'])-origin);mode='explore'
        elif image:
            retrieval=self.predict(prepared,'Localize target',image=target['image'],task='localization')
            photo,_=self.pose(retrieval,allow_absent=True)
            if photo is not None:
                direction=(rotation@Rotation.from_quat(photo[3:]).as_matrix()[:,0])[:2]
                if np.linalg.norm(direction)<1e-6:raise ValueError('Photo has no horizontal heading')
                goal=photo[:3]+rotation.T@np.r_[.75*direction/np.linalg.norm(direction),0.]
                extra['image_goal_forward_offset_m']=.75
        elif not job.get('force_explore'):
            retrieval=self.predict(prepared,target['text'],task='grounding',return_block_diagnostics=True)
            centers=np.asarray(retrieval.get('predicted_centers') or []).reshape(-1,3)
            if len(centers):
                counts=np.zeros(len(centers),dtype=np.int64)
                axes=rotation[:2,:].T
                for points,pred in zip(xyz,retrieval['block_predictions']):
                    tf=rigid_frame(pred); projected=(points@tf[:3,:3].T+tf[:3,3])@axes
                    for i,c in enumerate(centers@axes): counts[i]+=np.count_nonzero(np.sum((projected-c)**2,axis=1)<=.5**2)
                order,centers_world,distances=rank_retrieval_centers(centers,counts,rotation,origin,job.get('previous_goal_xy'))
                i=order[0];goal=centers[i]
                if self.model.metadata['joint_protocol']['version'].endswith('_v4'):
                    viewpoints = paired_viewpoints(retrieval.get('predicted_viewpoints'), centers)
                    viewpoint = viewpoints[i]
                    goal = viewpoint[:3]
                    heading = (rotation @ np.r_[viewpoint[3:], 0.])[:2]
                    norm = np.linalg.norm(heading)
                    extra.update(navigation_target_type='viewpoint', selected_viewpoint=viewpoint.tolist(),
                        navigation_goal_xyz=goal.tolist(), navigation_goal_world_xyz=(rotation @ goal + origin).tolist(),
                        viewpoint_heading_world_xy=(heading / norm).tolist() if norm > 1e-8 else None,
                        predicted_centers=centers.tolist(), predicted_viewpoints=viewpoints.tolist())
                mode='dialogue' if len(centers)==1 else 'dialogue-'+chr(65+i)
                extra.setdefault('navigation_target_type', 'v3_object_center')
                extra.update(selected_world_xy=centers_world[i,:2].tolist(),candidate_priority=0,candidate_point_counts=counts.tolist(),selected_center_xyz=centers[i].tolist())
        if goal is None:
            mode='explore'
            absence = retrieval['generated_text'].replace('<|im_end|>', '').strip() if image and retrieval else None
            messages = frontier_messages(pose[:3].tolist(), image_target=image, absence=absence)
            frontier=self.predict(prepared,'Localize' if image else target['text'],
                image=target['image'] if image else None,task='localization' if image else 'grounding',
                continuation_messages=messages)
            centers=np.asarray(frontier.get('predicted_centers') or [])
            if centers.shape!=(1,3) or not np.isfinite(centers).all() or frontier.get('predicted_poses') or frontier.get('trajectory'):raise ValueError('Frontier must return one position: '+str(frontier))
            goal=centers[0];extra['frontier']=frontier
        route=self.predict(prepared,'Navigate',image=job['image'],task='coordinate',
            goal_xyz=list(map(float,goal)),current_pose=pose.tolist(),localization_answer=current['answer'])
        trajectory=np.asarray(route.get('trajectory') or [])
        if trajectory.shape!=(80,3) or not np.isfinite(trajectory).all() or np.asarray(route.get('predicted_poses')).shape!=(1,7) or not np.isfinite(np.asarray(route.get('predicted_poses'))).all() or route.get('predicted_centers'):raise ValueError('Coordinate navigation must return 80 finite points')
        start=np.r_[rotation@pose[:3]+origin,(Rotation.from_matrix(rotation)*Rotation.from_quat(pose[3:])).as_quat()]
        result=dict(ok=True,mode=mode,label=mode,world_xyz=(trajectory@rotation.T+origin).tolist(),
            predicted_start_pose_world=start.tolist(),retrieval=retrieval,route=route,**extra)
        return dict(state='done',modes=[result],memories=job['memories'])
