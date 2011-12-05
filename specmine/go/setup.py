import os.path
import numpy

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

def numpy_include_dir():
    path = os.path.dirname(numpy.__file__)

    return os.path.join(path, "core/include")

setup(
  name = "PyGo",
  ext_modules=[ 
    Extension("gnugo_engine", ["gnugo_engine.pyx"],
              include_dirs=['gnugo-3.8','gnugo-3.8/sgf','gnugo-3.8/utils','gnugo-3.8/engine', numpy_include_dir()],
              extra_objects=[
                "gnugo-3.8/engine/libboard.a",
                "gnugo-3.8/engine/libengine.a",
                "gnugo-3.8/sgf/libsgf.a",
                "gnugo-3.8/utils/libutils.a",
                "gnugo-3.8/patterns/libpatterns.a"],
              library_dirs=['gnugo-3.8/engine','gnugo-3.8/sgf','gnugo-3.8/utils'],
              libraries = ["m","curses"]),
    
    Extension("go_loops", ["go_loops.pyx"],
              libraries = ["m","curses"]),
    ],
  cmdclass = {'build_ext': build_ext},
)



