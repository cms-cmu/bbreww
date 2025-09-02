import unittest
import yaml
import argparse
import sys

class TestCutflowCompare(unittest.TestCase):
    cutflow = None
    reference = None

    @classmethod
    def setUpClass(cls):
        # Files are set as class variables before unittest.main()
        pass

    def print_test_results(self, failures):
        print("\n")
        print(f'{"Sample":^30} {"Region":^15} {"Key":^10} {"Cut":^20} {"Observed":^15} {"Expected":^15} {"AbsDiff":^10}')
        for fail in failures:
            sample, region, key, cut, obs, exp = fail
            absdiff = round(float(obs) - float(exp), 5)
            print(f'{sample:^30} {region:^15} {key:^10} {cut:^20} {obs:^15} {exp:^15} {absdiff:^10}')
        print("\n")

    def test_cutflow_matches_yaml(self):
        failures = []
        for sample, sample_data in self.reference.items():
            self.assertIn(sample, self.cutflow)
            for region, region_data in sample_data['cutflow'].items():
                self.assertIn(region, self.cutflow[sample]['cutflow'])
                for key in ['events', 'weights']:
                    self.assertIn(key, region_data)
                    self.assertIn(key, self.cutflow[sample]['cutflow'][region])
                    for cut, value in region_data[key].items():
                        self.assertIn(cut, self.cutflow[sample]['cutflow'][region][key])
                        observed = self.cutflow[sample]['cutflow'][region][key][cut]
                        try:
                            self.assertAlmostEqual(
                                float(value),
                                float(observed),
                                places=5,
                                msg=f"Mismatch for {sample}/cutflow/{region}/{key}/{cut}"
                            )
                        except AssertionError:
                            failures.append((sample, region, key, cut, observed, value))
        if failures:
            self.print_test_results(failures)
        self.assertEqual(len(failures), 0, f"Found {len(failures)} mismatches in cutflow comparison.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare two YAML cutflow files')
    parser.add_argument('--input_file', required=True, help='Input YAML file')
    parser.add_argument('--known_cutflow', required=True, help='Reference YAML cutflow file')
    args, unknown = parser.parse_known_args()

    with open(args.input_file, 'r') as infile:
        cutflow = yaml.safe_load(infile)

    with open(args.known_cutflow, 'r') as yfile:
        reference = yaml.safe_load(yfile)

    TestCutflowCompare.cutflow = cutflow
    TestCutflowCompare.reference = reference

    sys.argv = sys.argv[:1] + unknown
    unittest.main()