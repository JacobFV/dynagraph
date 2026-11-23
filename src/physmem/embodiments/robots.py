"""Robot registry: every v2 embodiment by name. Asset provenance in docs/ROBOTS.md.

TCP convention (all robots): TCP z = approach direction (out of the palm/between the fingers), TCP y = finger
closing axis, origin at the centre between the finger pads. Mount convention: the robot's base is at the mount pose;
`front_yaw` rotates each model so that its home TCP lies along +x of the mount frame.
"""
from __future__ import annotations

from physmem.embodiments.core import EffectorCfg, GripperCfg, RobotCfg

MENAGERIE = 'mujoco_menagerie@f3b6013 (Apache-2.0 / BSD-3 per model)'
MENAGERIE_X = 'mujoco_menagerie@370c439 sparse (Apache-2.0 / BSD-3 per model)'
R90Z = (0.7071068, 0, 0, 0.7071068)     # TCP y = gripper x
RY90 = (0.7071068, 0, 0.7071068, 0)     # TCP z = body x


def _f2(name, joints, body, home, pre='g/'):
    return EffectorCfg(name=name, joints=joints, site_body=pre + 'base', tcp_pos=(0, 0, 0.145), closing=(0, 1, 0),
                       home=home, gripper=GripperCfg(kind='parallel', actuators=(pre + 'fingers_actuator',),
                                                     closed=(0.8,), opened=(0.0,), force=(5.0,), kd_ratio=0.0, max_width=0.085, tip=0.018, link_h=0.033, pad_per_drive=9.07, grip_n=100.0, pad_friction=1.2,
                                                     pad_bodies=tuple(pre + b for b in ('right_pad', 'left_pad', 'right_silicone_pad', 'left_silicone_pad', 'right_follower', 'left_follower'))))


ROBOTS = {
    # (a) Franka Emika Panda + Franka Hand (parallel)
    'panda': RobotCfg(
        name='panda', kind='arm', xml='menagerie_extra/franka_emika_panda/panda.xml',
        effectors=(EffectorCfg(name='hand', joints=tuple(f'joint{i}' for i in range(1, 8)), site_body='hand',
                               tcp_pos=(0, 0, 0.1034), home=(0, -0.1104, 0, -2.273, 0, 2.1626, -0.7854),
                               gripper=GripperCfg(kind='parallel', actuators=('actuator8',), closed=(0.0,), opened=(0.04,), max_width=0.08, tip=0.009, pad_per_drive=0.68, grip_n=70.0, pad_friction=1.2,
                                                  force=(70.0,), kd_ratio=0.0, pad_bodies=('left_finger', 'right_finger'))),),
        torque_override={'actuator8': (-280.0, 280.0)},
        reach_m=0.855, payload_kg=3.0, max_ee_speed=1.0, max_ee_force=70.0,
        limits=dict(v=1.2, a=8.0, j=200.0, w=3.0), source=MENAGERIE_X + ' franka_emika_panda', base_body='link0'),
    # (b) UR5e + Robotiq 2F-85
    'ur5e_2f85': RobotCfg(
        name='ur5e_2f85', kind='arm', xml='menagerie/universal_robots_ur5e/ur5e.xml',
        attach=(('attachment_site', 'menagerie_extra/robotiq_2f85/2f85.xml', 'g/'),),
        effectors=(_f2('hand', ('shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint',
                                'wrist_2_joint', 'wrist_3_joint'), 'g/base', (-1.8419, -1.5918, 1.6738, -1.6528, -1.5708, -1.8419)),),
        front_yaw=1.5708, home_target=(0.5, 0.0, 0.3),
        torque_override={'g/fingers_actuator': (-28.0, 28.0)},
        reach_m=0.85, payload_kg=5.0, max_ee_speed=1.0, max_ee_force=60.0,
        limits=dict(v=1.2, a=8.0, j=200.0, w=3.0), source=MENAGERIE + ' universal_robots_ur5e + ' + MENAGERIE_X + ' robotiq_2f85',
        base_body='base'),
    # (c) Kinova Gen3 + Robotiq 2F-85
    'gen3_2f85': RobotCfg(
        name='gen3_2f85', kind='arm', xml='menagerie_extra/kinova_gen3/gen3.xml',
        attach=(('pinch_site', 'menagerie_extra/robotiq_2f85/2f85.xml', 'g/'),),
        effectors=(_f2('hand', tuple(f'joint_{i}' for i in range(1, 8)), 'g/base',
                       (0.0, 0.487, 3.143, -1.353, 0.0, -1.302, 1.574)),),
        torque_override={'g/fingers_actuator': (-28.0, 28.0)},
        reach_m=0.9, payload_kg=4.0, max_ee_speed=0.8, max_ee_force=50.0,
        limits=dict(v=1.0, a=6.0, j=150.0, w=2.5), source=MENAGERIE_X + ' kinova_gen3 + robotiq_2f85', base_body='base_link'),
    # (c') Unitree Z1 with its single-jaw gripper
    'z1': RobotCfg(
        name='z1', kind='arm', xml='menagerie_extra/unitree_z1/z1_gripper.xml',
        effectors=(EffectorCfg(name='hand', joints=tuple(f'joint{i}' for i in range(1, 7)), site_body='link06',
                               tcp_pos=(0.186, 0, -0.009), approach=(1, 0, 0), closing=(0, 0, 1), home=(0, 1.9, -1.85, 1.45, 0, 0),
                               gripper=GripperCfg(kind='parallel', actuators=('motorGripper',), closed=(0.0,), opened=(-1.0,), jaw_dir=1, max_width=0.055, tip=0.014, pad_per_drive=25.0, grip_n=80.0, pad_friction=1.2,
                                                  force=(8.0,), kd_ratio=0.0, pad_bodies=('gripperMover', 'link06'))),),
        joint_overrides={f'joint{i}': dict(armature=0.02) for i in range(1, 7)},
        home_target=(0.3, 0.0, 0.25),
        reach_m=0.74, payload_kg=2.0, max_ee_speed=0.8, max_ee_force=30.0,
        limits=dict(v=1.0, a=6.0, j=150.0, w=3.0), source=MENAGERIE_X + ' unitree_z1', base_body='link00'),
    # (d) low-cost arm: TRS SO-ARM100 (5 dof + jaw)
    'so100': RobotCfg(
        name='so100', kind='arm', xml='menagerie_extra/trs_so_arm100/so_arm100.xml',
        effectors=(EffectorCfg(name='hand', joints=('Rotation', 'Pitch', 'Elbow', 'Wrist_Pitch', 'Wrist_Roll'),
                               site_body='Fixed_Jaw', tcp_pos=(0.011, -0.092, 0), approach=(0, -1, 0), closing=(1, 0, 0), home=(0.0071, -1.9069, 1.904, 1.5737, 1.5637),
                               rot_weight=0.05,
                               gripper=GripperCfg(kind='parallel', actuators=('Jaw',), closed=(-0.15,), opened=(1.0,), jaw_dir=-1, max_width=0.045, tip=0.015, pad_per_drive=15.0, grip_n=40.0, pad_friction=1.2,
                                                  force=(3.0,), kd_ratio=0.0, pad_bodies=('Fixed_Jaw', 'Moving_Jaw'))),),
        front_yaw=-1.5708, home_target=(0.2, 0.0, 0.1),
        reach_m=0.35, payload_kg=0.25, max_ee_speed=0.5, max_ee_force=8.0,
        limits=dict(v=0.6, a=4.0, j=80.0, w=3.0), gains=dict(kp=600.0, ki=1500.0),
        source=MENAGERIE_X + ' trs_so_arm100', base_body='Base'),
    # (e) bimanual: ALOHA (two ViperX 300s)
    'aloha': RobotCfg(
        name='aloha', kind='bimanual', xml='menagerie_extra/aloha/aloha.xml',
        effectors=tuple(EffectorCfg(name=s, joints=tuple(f'{s}/{j}' for j in ('waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate')),
                                    site_body=f'{s}/gripper_link', tcp_pos=(0.13, 0, -0.003), approach=(1, 0, 0), closing=(0, 1, 0),
                                    home=(0.0642, -0.3036, 0.6484, 0, 1.226, 1.635) if s == 'left' else (-0.0642, -0.3036, 0.6484, 0, 1.226, -1.635),
                                    gripper=GripperCfg(kind='parallel', actuators=(f'{s}/gripper',), closed=(0.0,), opened=(0.037,), max_width=0.065, tip=0.016,
                                                       force=(20.0,), kd_ratio=0.0, pad_bodies=(f'{s}/left_finger_link', f'{s}/right_finger_link')))
                        for s in ('left', 'right')),
        reach_m=0.6, payload_kg=0.75, max_ee_speed=0.8, max_ee_force=20.0,
        limits=dict(v=1.0, a=6.0, j=150.0, w=3.0), source=MENAGERIE_X + ' aloha', base_body=''),
    # (g) mobile manipulator: Stanford TidyBot (holonomic base + Gen3 + 2F-85)
    'tidybot': RobotCfg(
        name='tidybot', kind='mobile_manipulator', xml='menagerie/stanford_tidybot/tidybot.xml',
        effectors=(_f2('hand', tuple(f'joint_{i}' for i in range(1, 8)), 'base',
                       (-0.4756, 0.3033, 3.8863, -1.1031, -0.2098, -1.8072, 1.7669), pre=''),),   # TCP 0.78 m: above a 0.76 m bench
        base_joints=('joint_x', 'joint_y', 'joint_th'),
        torque_override={'joint_x': (-800, 800), 'joint_y': (-800, 800), 'joint_th': (-300, 300), 'fingers_actuator': (-28.0, 28.0)},
        home_target=(0.42, 0.0, 0.78),
        reach_m=0.9, payload_kg=4.0, max_ee_speed=0.8, max_ee_force=50.0,
        limits=dict(v=1.0, a=6.0, j=150.0, w=2.5), source=MENAGERIE + ' stanford_tidybot', base_body='base_link'),
}


def _suction_eff(joints, home):
    return EffectorCfg(name='hand', joints=joints, site_body='cup', tcp_pos=(0, 0, 0.078), home=home,
                       gripper=GripperCfg(kind='suction', actuators=('suction',), max_width=1.0, tip=0.0, link_h=0.07,
                                          pad_bodies=('cup',), close_time=0.1))


SUCTION = dict(kind='suction', site='attachment_site', stem=0.07, radius=0.035, gain=300.0)   # 70 mm foam cup, ~-0.8 bar
ROBOTS['ur5e_suction'] = RobotCfg(
    name='ur5e_suction', kind='arm', xml='menagerie/universal_robots_ur5e/ur5e.xml', tool=SUCTION,
    effectors=(_suction_eff(('shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint',
                             'wrist_3_joint'), (-1.8419, -1.5918, 1.6738, -1.6528, -1.5708, -1.8419)),),
    front_yaw=1.5708, reach_m=0.85, payload_kg=5.0, max_ee_speed=1.0, max_ee_force=60.0,
    limits=dict(v=1.2, a=8.0, j=200.0, w=3.0), source=MENAGERIE + ' universal_robots_ur5e + vacuum cup (adhesion)', base_body='base')
ROBOTS['fr3_suction'] = RobotCfg(
    name='fr3_suction', kind='arm', xml='menagerie/franka_fr3/fr3.xml', tool=SUCTION,
    effectors=(_suction_eff(tuple(f'fr3_joint{i}' for i in range(1, 8)), (0, -0.1104, 0, -2.273, 0, 2.1626, -0.7854)),),
    reach_m=0.855, payload_kg=3.0, max_ee_speed=1.0, max_ee_force=70.0,
    limits=dict(v=1.2, a=8.0, j=200.0, w=3.0), source=MENAGERIE + ' franka_fr3 + vacuum cup (adhesion)', base_body='base')


def names():
    return sorted(ROBOTS) + ['g1_dex3', 'g1_dex3_fixed']
