#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import glob

SUFFIX = ".invert"


def invert_line(line):
    """
    :arg line utf-8 encoded sentence
    """

    return line.decode("utf-8")[::-1].encode("utf-8")


def invert_file(filename):
    with open(filename, "r") as fin, open(filename + SUFFIX, "w") as fou:
        lines = []
        for line in fin:
            lines.append(invert_line(line.strip()))
        fou.write('\n'.join(lines))


if __name__ == "__main__":
    files = glob.glob("../working_data/*")
    for fn in files:
        print("inverting filename: " + fn)
        invert_file(fn)
