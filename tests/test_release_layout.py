import importlib.util, pathlib, unittest, json, hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('preflight',ROOT/'scripts/preflight_release.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
class ReleaseLayoutTests(unittest.TestCase):
 def test_seven_locked_source_files(self):
  d=json.loads((ROOT/'delivery/r2b/source/accepted_sources.lock.json').read_text())
  self.assertEqual(len(d['files']),7)
  for rel,meta in d['files'].items():
   source=ROOT/'delivery/r2b/source'/rel
   self.assertEqual(source.stat().st_size,meta['bytes'])
   self.assertEqual(p.sha(source),meta['sha256'])
 def test_no_private_repository_dependency(self):
  nb=(ROOT/'notebooks/SolarSpine_R2B_Jurado_T4.ipynb').read_text()
  self.assertNotIn('PACKAGE_ZIP',nb)
  self.assertNotIn('MyDrive/SolarSpine',nb)
  self.assertIn('git',nb)
 def test_release_gate_prepublication(self):
  s=p.verify(False)
  self.assertEqual(s['status'],'PASS')
  self.assertFalse(s['release_ready'])
 def test_two_checkpoint_locks(self):
  self.assertEqual(set(p.EXPECTED),{'models/detector/best.pt','models/refiner/best.pt'})
 def test_runner_syntax(self):
  compile((ROOT/'delivery/r2b/run_inference.py').read_bytes(),'run_inference.py','exec')
  compile((ROOT/'delivery/r2b/source/scripts/r2b_replay_backend.py').read_bytes(),'r2b_replay_backend.py','exec')
 def test_notebook_syntax(self):
  cells=json.loads((ROOT/'notebooks/SolarSpine_R2B_Jurado_T4.ipynb').read_text())['cells']
  for i,c in enumerate(cells):
   if c['cell_type']=='code':compile(''.join(c['source']),f'cell{i}','exec')
if __name__=='__main__':unittest.main()
