"""Non-mutating release audit for the SolarSpine public jury repository."""
from pathlib import Path
import json, subprocess, sys
R=Path(__file__).resolve().parents[1]
required=['THIRD_PARTY_NOTICE.md','PUBLISHING.md','docs/MODEL_PROVENANCE.md',
          'LICENSE-MIT-ORIGINAL','notebooks/SolarSpine_R2B_Jurado_T4.ipynb',
          'models/detector/best.pt','models/refiner/best.pt']
checks={f: (R/f).is_file() for f in required}
license_text=(R/'LICENSE').read_text(encoding='utf-8',errors='replace')
checks['AGPL_3_FULL_LICENSE']=len(license_text)>25000 and 'GNU AFFERO GENERAL PUBLIC LICENSE' in license_text and 'Version 3, 19 November 2007' in license_text
checks['MIT_HISTORICAL']= (R/'LICENSE-MIT-ORIGINAL').read_text(encoding='utf-8',errors='replace').startswith('MIT License') if checks['LICENSE-MIT-ORIGINAL'] else False
# Check tracked paths that should never be part of this public release.
try:
    r=subprocess.run(['git','ls-files'],cwd=R,capture_output=True,text=True,check=True)
    paths=r.stdout.splitlines()
except Exception:
    paths=[]  # standalone unpack test; path review remains user's responsibility
block=['.env','kaggle.json','gcloud','secret','credentials.json', '.pem','id_rsa', 'test_images/', 'train_images/', 'test_masks/']
flag=[p for p in paths if any(x in p.lower() for x in block) and not p.startswith('scripts/check_publication_ready')]
checks['NO_SUSPICIOUS_TRACKED_PATHS']=not flag
print(json.dumps({'checks':checks,'suspicious_tracked_paths':flag,
                  'does_not_check':'Git history, semantic secrets, third-party rights or Kaggle organizer approval'},indent=2,ensure_ascii=False))
if not all(checks.values()): sys.exit('PUBLICATION_PREFLIGHT_NOT_READY')
print('PUBLICATION_FILES_GATE_PASS__RIGHTS_AND_ANONYMOUS_RUNTIME_STILL_REQUIRE_REVIEW')
