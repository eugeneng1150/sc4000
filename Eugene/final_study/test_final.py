"""Integrity and no-training replay tests for the frozen final notebook."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
import final_report as report

HERE=Path(__file__).resolve().parent
class FinalTests(unittest.TestCase):
    def test_replay_preserves_exact_submission_and_is_repeatable(self):
        payload=(HERE/'frozen_evidence.zip').read_bytes()
        with tempfile.TemporaryDirectory() as t:
            d=Path(t);e=d/'evidence'
            manifest=report.unpack(payload,report.sha(payload),e)
            self.assertEqual(report.validate_submission(e)['rows'],894)
            path=report.export_submission(e,d)
            self.assertEqual(report.sha(path.read_bytes()),manifest['submission_sha256'])
            report.unpack(payload,report.sha(payload),e)
            self.assertEqual(report.export_submission(e,d).read_bytes(),path.read_bytes())
            path.write_text('different submission')
            with self.assertRaisesRegex(ValueError,'different submission'):report.export_submission(e,d)

    def test_tampered_package_or_submission_rejected(self):
        payload=(HERE/'frozen_evidence.zip').read_bytes()
        with tempfile.TemporaryDirectory() as t:
            d=Path(t)
            with self.assertRaisesRegex(ValueError,'package hash'):report.unpack(payload+b'x',report.sha(payload),d)
            report.unpack(payload,report.sha(payload),d)
            (d/'submission_selected.csv').write_text('ID,TopConfigs\nx,0\n')
            with self.assertRaisesRegex(ValueError,'submission changed'):report.validate_submission(d)
            with self.assertRaisesRegex(ValueError,'Existing evidence differs'):report.unpack(payload,report.sha(payload),d)

    def test_model_archive_streaming_hash_checks(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t);(d/'tables').mkdir();data=b'synthetic model bytes'
            (d/'tables/model_inventory.csv').write_text('path,sha256\nmodel.pt,'+report.sha(data)+'\n')
            archive=d/'models.zip'
            with zipfile.ZipFile(archive,'w') as z:z.writestr('model.pt',data)
            self.assertIn('Verified 1',report.verify_model_archive(archive,d))
            with zipfile.ZipFile(archive,'w') as z:z.writestr('model.pt',data+b'changed')
            with self.assertRaisesRegex(ValueError,'Model differs'):report.verify_model_archive(archive,d)

    def test_notebook_embeds_current_code_and_evidence(self):
        import nbformat
        notebook=nbformat.read(HERE.parent/'SC4000_Final.ipynb',as_version=4)
        nbformat.validate(notebook)
        code='\n'.join(c.source for c in notebook.cells if c.cell_type=='code')
        for c in notebook.cells:
            if c.cell_type=='code':ast.parse(c.source)
        self.assertIn(repr((HERE/'final_report.py').read_text()),code)
        self.assertIn(report.sha((HERE/'frozen_evidence.zip').read_bytes()),code)
        # Plot/export code contains no model loading, training or shell execution.
        for token in ['.fit(', 'joblib.load(', 'torch.load(', 'subprocess.', 'run_study(']:
            self.assertNotIn(token,(HERE/'final_report.py').read_text())

if __name__=='__main__':unittest.main(verbosity=2)
