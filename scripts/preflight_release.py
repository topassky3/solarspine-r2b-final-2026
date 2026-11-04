"""Standard-library source/weights release gate for SolarSpine R2B.

Does not import ML packages or contact a private service. Not a GPU inference test.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
LOCK=ROOT/'delivery/r2b/source/accepted_sources.lock.json'
SOURCE=ROOT/'delivery/r2b/source'
EXPECTED={
 'models/detector/best.pt':'bbb3c55175f68f80c213ac9fd6d1a1ab5bd63993696b609348d36f0472515875',
 'models/refiner/best.pt':'e4b9c3bacce816cd1d4dfb863622b93774743a6a401ae844088e33540882f397',
}
REQUIRED=['README.md','requirements.txt','notebooks/SolarSpine_R2B_Jurado_T4.ipynb',
          'delivery/r2b/run_inference.py','delivery/r2b/source/scripts/r2b_replay_backend.py']

def sha(p):
 d=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):d.update(b)
 return d.hexdigest()

def verify(strict=False):
 result={'sources':{},'checkpoints':{},'files':{},'strict':strict}
 for p in REQUIRED:
  result['files'][p]=(ROOT/p).is_file()
 if not LOCK.is_file(): raise RuntimeError('Source lock absent')
 lock=json.loads(LOCK.read_text())
 if lock.get('schema')!='R2B-accepted-production-sources-v1' or len(lock.get('files',{}))!=7:
  raise RuntimeError('Original source lock schema or count changed')
 for rel,v in lock['files'].items():
  p=(SOURCE/rel).resolve()
  if not p.is_relative_to(SOURCE.resolve()): raise RuntimeError('Source path traversal: '+rel)
  ok=p.is_file() and p.stat().st_size==v['bytes'] and sha(p)==v['sha256']
  result['sources'][rel]=ok
 for rel,expected in EXPECTED.items():
  p=ROOT/rel
  result['checkpoints'][rel]=('PASS' if p.is_file() and sha(p)==expected else
                               'MISSING' if not p.exists() else 'SHA_MISMATCH')
 # Structural checks against a private-ZIP dependency in the jury notebook.
 nb=ROOT/'notebooks/SolarSpine_R2B_Jurado_T4.ipynb'
 if nb.is_file():
  obj=json.loads(nb.read_text())
  source=''.join(''.join(c.get('source',[])) for c in obj['cells'])
  result['notebook_repo_clone_ready']=('git clone' in source or "'clone'" in source) and 'PACKAGE_ZIP' not in source
  result['notebook_no_private_drive']='MyDrive/SolarSpine' not in source and 'PERSONAL_ACCESS_TOKEN' not in source
 else:
  result['notebook_repo_clone_ready']=False;result['notebook_no_private_drive']=False
 code_ok=all(result['sources'].values()) and all(result['files'].values()) and result['notebook_repo_clone_ready'] and result['notebook_no_private_drive']
 weights_ok=all(x=='PASS' for x in result['checkpoints'].values())
 result['status']='PASS' if code_ok and (weights_ok or not strict) else 'FAIL'
 result['release_ready']=code_ok and weights_ok
 return result

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--strict',action='store_true',help='Require both verified checkpoints in repo')
 args=p.parse_args()
 r=verify(args.strict)
 print(json.dumps(r,indent=2,sort_keys=True))
 if r['status']!='PASS':raise SystemExit(1)
 print('SOLARSPINE_RELEASE_SOURCE_GATE_PASS__WEIGHTS_'+('PASS' if r['release_ready'] else 'PENDING'))
if __name__=='__main__':main()
