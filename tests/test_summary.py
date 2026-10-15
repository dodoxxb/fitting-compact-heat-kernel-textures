"""Meaningful regression: failures, exact images, duplicate endpoints and cohort separation."""
import importlib.util
from pathlib import Path
import unittest

source = Path(__file__).resolve().parents[1] / 'tools/summarize_records.py'
spec = importlib.util.spec_from_file_location('summary', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SummaryTests(unittest.TestCase):
    def data(self):
        objects = [dict(uid=str(i), cohort=c) for i,c in enumerate(['public','public','external','external'])]
        rows = []
        for i,obj in enumerate(objects):
            for name,value in [('HKTex', 40.), ('warm3000', 40.2)]:
                rows.append(dict(**obj, seed=0, method=name, valid_model=True, sources=5000 if name=='HKTex' else 3000,
                                 metrics={'mean_view_psnr': value}))
        rows[3]['valid_model'] = False
        rows[5]['metrics']['mean_view_psnr'] = None
        return dict(objects=objects, records=rows,
                    comparisons=[dict(method='warm3000', reference='HKTex', budget=3000)])

    def test_failed_and_exact_views_are_explicit(self):
        result = module.summarize(self.data(), 'mean_view_psnr')
        self.assertEqual(len(result['paired']), 2)
        self.assertEqual(len(result['missing']), 2)
        self.assertEqual([row['objects'] for row in result['seed0_summary']], [1,1])

    def test_duplicate_endpoint_is_rejected(self):
        data = self.data()
        data['records'].append(data['records'][0])
        with self.assertRaises(ValueError):
            module.summarize(data, 'mean_view_psnr')

    def test_parent_budget_eligibility(self):
        data = self.data()
        data['records'][0]['sources'] = 999
        result = module.summarize(data, 'mean_view_psnr')
        self.assertIn('budget_ineligible', [r['reason'] for r in result['missing']])


if __name__ == '__main__':
    unittest.main()
