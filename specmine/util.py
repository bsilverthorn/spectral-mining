import os.path
import bz2
import gzip
import contextlib

def openz(path, mode = "rb", closing = True):
    """Open a file, transparently [de]compressing it if a known extension is present."""

    (_, extension) = os.path.splitext(path)

    if extension == ".bz2":
        file_ = bz2.BZ2File(path, mode)
    elif extension == ".gz":
        file_ = gzip.GzipFile(path, mode)
    elif extension == ".xz":
        raise NotImplementedError()
    else:
        return open(path, mode)

    if closing:
        return contextlib.closing(file_)

def static_path(relative):
    """Return the path to an associated static data file."""

    return os.path.join(os.path.dirname(__file__), "static", relative)

