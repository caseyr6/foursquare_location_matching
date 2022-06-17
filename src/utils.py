from typing import Union
from pathlib import Path


def _file_len(
    fname: Union[str, Path],
):
    """
    Given a text file, return the number of lines in that file.

    Parameters
    ----------
    fname: `str or Path-like`
        Name of the file. This should be the absolute file name.

    Returns
    -------
    Number of lines in the file.
    """
    with open(fname) as f:
        for i, l in enumerate(f):
            pass

        if 'i' not in locals():
            i = 0
    return i + 1


def _file_del_n_lines(
    fname: Union[str, Path],
    n_lines: int,
):
    """
    Given a text file, only keep the specified number of lines.

    Parameters
    ----------
    fname: `str or Path-like`
        Name of file to remove lines from. This should be the absolute file
        name.
    n_lines: `int`
        Number of lines to keep in the file.
    """
    f = open(fname, 'r')
    lines = f.readlines()
    f.close()

    del lines[:n_lines]

    f = open(fname, 'w')
    f.writelines(lines)
    f.close()
    del lines