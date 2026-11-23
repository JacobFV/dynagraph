"""render the robot at the pre-grasp of a pick (gripper open) from two views -> results/v2/robots/snap_<robot>.png"""
import sys, os
import numpy as np, mujoco
from physmem.v2.api import SkillCall
from physmem.embodiments import testbench as TB
from physmem.skills import SkillContext
from physmem.skills import primitives as P
robot = sys.argv[1]
B = TB.BENCH[robot]
half = (B['obj'][1],) * 3
(x0, x1), (y0, y1) = B['ws']
pos = ((x0 + x1) / 2, (y0 + y1) / 2, B['table_z'] + half[2])
spec, emb, bench = TB.make_scene(robot, extra=lambda s, b: TB.add_box(s, 'obj', pos, half, 0.1))
m, d = TB.compile_scene(spec, emb)
eff = next(iter(emb.eff))
ctx = SkillContext(m, d, emb)
from physmem.embodiments.ik import down_quat
q = down_quat(P._near_yaw(ctx, eff, 0.0))
P._move(ctx, eff, [np.array([pos[0], pos[1], B['table_z'] + 2 * half[2] + 0.04])], 0.3, q, gripper=0.0)
ctx.wait(0.5)
r = mujoco.Renderer(m, 480, 640)
imgs = []
for az, el in ((90, -10), (0, -10)):
    cam = mujoco.MjvCamera(); cam.lookat[:] = pos; cam.distance = 0.6 if robot != 'so100' else 0.35; cam.azimuth = az; cam.elevation = el
    r.update_scene(d, cam); imgs.append(r.render())
os.makedirs('results/v2/robots', exist_ok=True)
import imageio.v2 as iio
iio.imwrite(f'results/v2/robots/snap_{robot}.png', np.concatenate(imgs, 1))
p, qq = emb.tcp_pose(d, eff)
print('tcp', p.round(3), 'z axis', TB.quat2mat(qq)[:, 2].round(2) if hasattr(TB, 'quat2mat') else '', 'grip', emb.gripper_state(m, d, eff))
