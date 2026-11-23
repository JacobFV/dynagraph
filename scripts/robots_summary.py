"""collect results/v2/robots/<robot>/*.json into a markdown table (stdout) + results/v2/robots/summary.json."""
import json, os

OUT = 'results/v2/robots'
ROBOTS = ['panda', 'ur5e_2f85', 'gen3_2f85', 'z1', 'so100', 'aloha', 'tidybot', 'ur5e_suction', 'fr3_suction']


def load(r, t):
    p = f'{OUT}/{r}/{t}.json'
    return json.load(open(p))['summary'] if os.path.exists(p) else None


def f(x, k='p50', s=1.0, nd=3):
    if not x or k not in x:
        return '-'
    return f'{x[k] * s:.{nd}f}'


rows, allj = [], {}
hdr = ('| robot | mid-stroke err p50/p95/max (mm) | speed ratio p10/p50/p90 | release ratio p10/p90 | stalls /50 | '
       'time at torque limit (s) | peak torque-rate p50/p90 (1/s) | peak joint jerk p50/p90 (rad/s^3) | sim/wall | '
       'pick+place /20 | place err p50 (mm) | push ratio p10/p50/p90 | sweep 30 parts moved | insert 1.0 / 0.5 mm | integrity |')
print(hdr)
print('|' + '---|' * hdr.count('|')[:-1] if False else '|' + '|'.join(['---'] * (hdr.count('|') - 1)) + '|')
for r in ROBOTS:
    s, pp, pu, sw, ins, ho = (load(r, t) for t in ('strokes', 'pickplace', 'push', 'sweep', 'insert', 'handover'))
    allj[r] = dict(strokes=s, pickplace=pp, push=pu, sweep=sw, insert=ins, handover=ho)
    if s is None:
        continue
    e = s['err_mid_max']
    integ = (len(s.get('integrity', [])) + (pp or {}).get('integrity_violations', 0) + (pu or {}).get('integrity_violations', 0)
             + (sw or {}).get('integrity_violations', 0))
    ins_s = '-' if not ins else f"{ins.get('1.0mm', {}).get('success', 0) * 10:.0f}/10, {ins.get('0.5mm', {}).get('success', 0) * 10:.0f}/10"
    print(f"| {r} | {f(e, 'p50', 1e3, 1)}/{f(e, 'p95', 1e3, 1)}/{f(e, 'max', 1e3, 1)} | "
          f"{f(s['ratio'], 'p10')}/{f(s['ratio'], 'p50')}/{f(s['ratio'], 'p90')} | {f(s['release_ratio'], 'p10')}/{f(s['release_ratio'], 'p90')} | "
          f"{s['stalls']} | {s['time_at_limit_s']:.2f} | {f(s['peak_torque_rate'], 'p50', 1, 1)}/{f(s['peak_torque_rate'], 'p90', 1, 1)} | "
          f"{f(s['peak_joint_jerk'], 'p50', 1, 0)}/{f(s['peak_joint_jerk'], 'p90', 1, 0)} | {s['rt_factor']:.1f}x | "
          f"{'-' if not pp else round(pp['success'] * pp['n'])} | {f((pp or {}).get('err_xy'), 'p50', 1e3, 1)} | "
          f"{'-' if not pu else f(pu['speed_ratio'], 'p10') + '/' + f(pu['speed_ratio'], 'p50') + '/' + f(pu['speed_ratio'], 'p90')} | "
          f"{'-' if not sw else f(sw['frac_moved'], 'p50', 100, 0) + '%'} | {ins_s} | {integ} |")
    if ho:
        print(f'|   handover ({r}) | {round(ho["success"] * ho["n"])}/{ho["n"]} successful |' + ' |' * 13)
json.dump(allj, open(f'{OUT}/summary.json', 'w'), indent=1, default=float)
fails = {r: (allj[r]['pickplace'] or {}).get('failures') for r in ROBOTS}
print('\npick/place failure modes:', json.dumps(fails))
