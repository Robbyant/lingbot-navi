"""Preserve a source camera's pitch without tilting the navigation body."""
import math


def set_initial_sensor_pitch(agent,pitch_rad):
    pitch=float(pitch_rad)
    if not math.isfinite(pitch) or abs(pitch)>=math.pi/2:
        raise ValueError('Initial source sensor pitch must be finite and within +/-90 degrees')
    if abs(pitch)<=1e-12:return
    import quaternion
    state=agent.get_state()
    if not state.sensor_states:raise ValueError('Initial agent has no sensor states')
    local_pitch=quaternion.from_rotation_vector([pitch,0.,0.])
    for sensor in state.sensor_states.values():sensor.rotation=sensor.rotation*local_pitch
    agent.set_state(state,reset_sensors=False,infer_sensor_states=False)
