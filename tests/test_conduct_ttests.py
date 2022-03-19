#!/usr/bin/env python

# import modules
import unittest

import pandas as pd

from panseer.panseer import conduct_ttests


class TestConductTtest(unittest.TestCase):
    def test_basic(self):
        df = pd.DataFrame(
            {
                'sample1':[1,1,1,1,1],
                'sample2':[2,2,2,2,2],
                'sample3':[3,3,3,3,3]
            },
            index=['marker1','marker2','marker3','marker4','marker5']
        )