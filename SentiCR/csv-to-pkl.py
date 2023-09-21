#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
__author__ = "Justin Ribeiro <justin@justinribeiro.com>"

import argparse
import csv
import pickle
import pandas as pd
import sys

csv.field_size_limit(sys.maxsize)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSV Pump into Pickle File")

    parser.add_argument(
        "--input", type=str, help="CSV file",
    )
    parser.add_argument(
        "--output", type=str, help="PKL file",
    )

    args = parser.parse_args()
    inputFile = args.input
    outputFile = args.output

df = pd.read_csv(inputFile, header=0)
df.to_pickle(outputFile)
