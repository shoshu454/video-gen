# Summarise rendered walkthrough items of one template into <OUT_ROOT>/walkthrough_<T>/items.md (a table built
# from each item<k>/meta.json), for the hand-written README.md next to it.
#   python walkthrough_index.py T3
import json, os, sys
from common import OUT_ROOT

tpl = sys.argv[1]
root = f'{OUT_ROOT}/walkthrough_{tpl}'
rows = []
for it in sorted(x for x in os.listdir(root) if x.startswith('item') and os.path.exists(f'{root}/{x}/meta.json')):
    m = json.load(open(f'{root}/{it}/meta.json'))
    ck = m['checks']
    if tpl == 'T3':
        pts = '; '.join(f"{L}: track {p['track_index']}, {p['region']}{' (moving)' if p['on_moving_agent'] else ''}"
                        for L, p in m['points'].items())
        gt = (f"overlap on baked frames {m['overlap_baked_frames']}" if m['overlap_baked_frames'] else
              f"closest approach {min(m['dist_2d_baked_px']):.0f} px (radius {m['overlap_radius_px']:.1f} px)")
        rows.append(f"| `{it}` | {m['kind']} | `{m['video_id']}` | {m['window'][0]}–{m['window'][1]} | {pts} | {gt} | "
                    f"**{m['answer']}** | {'✓' if ck['label_consistent'] and ck['ring_only_on_first_3'] else '✗'} |")
    else:
        pts = '; '.join(f"{L}: r {p['r_t0']:.1f} → {p['r_te']:.1f} m" for L, p in m['points'].items())
        rows.append(f"| `{it}` | {m['event_text']} | `{m['video_id']}` | {m['window'][0]}–{m['window'][1]}, t_e = {m['t_e']} | "
                    f"{pts} | {m['closest_t0']} → **{m['closest_te']}** | "
                    f"{', '.join(f'{k} {v:.0%}' for k, v in m['margins_te'].items())} | "
                    f"{'✓' if ck['label_consistent'] and ck['ring_only_on_first_3'] else '✗'} |")
if tpl == 'T3':
    head = '| item | kind | video | window (frames) | points | GT | answer | checks |\n|---|---|---|---|---|---|---|---|'
else:
    head = ('| item | event | drive | window, event frame | points (t0 → t_e) | closest t0 → t_e | margins at t_e | checks |\n'
            '|---|---|---|---|---|---|---|---|')
open(f'{root}/items.md', 'w').write(head + '\n' + '\n'.join(rows) + '\n')
print(head + '\n' + '\n'.join(rows))
