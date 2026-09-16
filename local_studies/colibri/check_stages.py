"""Run consecutive body-prefix stability checks, stopping at the first failure."""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import warp as wp
from newton.examples.kamino.example_kamino_colibri import BODY_ORDER, Example
from newton.viewer import ViewerNull

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, default=1)
parser.add_argument('--end', type=int, default=len(BODY_ORDER))
parser.add_argument('--frames', type=int, default=180)
args = parser.parse_args()
results = []
wp.init()
for n in range(args.start, args.end+1):
    start = time.perf_counter()
    example = Example(ViewerNull(), argparse.Namespace(body_count=n))
    peak_linear = peak_angular = 0.0
    peak_contacts = 0
    for frame in range(args.frames):
        example.step()
        example.test_post_step()
        peak_contacts = max(peak_contacts, int(example.solver._contacts_kamino.model_active_contacts.numpy()[0]))
        assert peak_contacts < example.contact_capacity, "Contact buffer full"
        qd = example.state_0.body_qd.numpy()
        peak_linear = max(peak_linear, float(np.max(np.linalg.norm(qd[:,:3], axis=1))))
        peak_angular = max(peak_angular, float(np.max(np.linalg.norm(qd[:,3:], axis=1))))
    example.test_final()
    result = dict(bodies=n, added=BODY_ORDER[n-1], frames=args.frames, peak_linear=peak_linear,
                  peak_angular=peak_angular, peak_contacts=peak_contacts, seconds=time.perf_counter()-start)
    results.append(result)
    print('PASS',json.dumps(result),flush=True)
    Path(f'/tmp/colibri_stages_{args.start}_{args.end}.json').write_text(json.dumps(results,indent=2))
    del example
    gc.collect()
