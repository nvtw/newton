
import sys
import json
from pathlib import Path
from local_studies.colibri.validate_phoenx import main
for body in range(9,15):
    output=Path(f"/tmp/colibri_reduced_classic_source_prefix{body}_20260915.json")
    sys.argv=["validate_phoenx","--body-count",str(body),"--substeps","10","--frames","30","--output",str(output)]
    try:
        main()
    except SystemExit as error:
        if error.code:
            print("FIRST_FAILED_PREFIX",body,flush=True)
            break
    if json.loads(output.read_text())["status"]!="passed":
        print("FIRST_FAILED_PREFIX",body,flush=True)
        break
