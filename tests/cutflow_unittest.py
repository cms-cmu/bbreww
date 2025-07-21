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

    def test_cutflow_matches_yaml(self):
        for sample, sample_data in self.reference.items():
            self.assertIn(sample, self.cutflow)
            for channel, channel_data in sample_data.items():
                self.assertIn(channel, self.cutflow[sample])
                for category, category_data in channel_data.items():
                    self.assertIn(category, self.cutflow[sample][channel])
                    for cut, value in category_data.items():
                        self.assertIn(cut, self.cutflow[sample][channel][category])
                        self.assertAlmostEqual(
                            float(value),
                            float(self.cutflow[sample][channel][category][cut]),
                            places=5,
                            msg=f"Mismatch for {sample}/{channel}/{category}/{cut}"
                        )

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