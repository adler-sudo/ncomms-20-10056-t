#!/usr/bin/env python

# import modules
import unittest

from panseer.panseer import combine_training_dataset_elementwise


# build
class TestCombineTrainingDatasetElementwise(unittest.TestCase):
    def test_short(self):
        combined_correct = [(1,1,1),(2,2,2),(3,3,3)]
        a = [1,2,3]
        b = [1,2,3]
        c = [1,2,3]
        combined = combine_training_dataset_elementwise(a,b,c)
        self.assertEqual(combined, combined_correct)
