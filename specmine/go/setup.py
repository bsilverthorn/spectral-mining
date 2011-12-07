import os.path
import numpy
import distutils.core
import distutils.extension
import Cython.Distutils

def numpy_include_dir():
    path = os.path.dirname(numpy.__file__)

    return os.path.join(path, "core/include")

distutils.core.setup(
    name = "PyGo",
    ext_modules=[ 
        #distutils.extension.Extension(
            #"gnugo_engine",
            #["gnugo_engine.pyx"],
            #include_dirs = [
                #'gnugo-3.8',
                #'gnugo-3.8/sgf',
                #'gnugo-3.8/utils',
                #'gnugo-3.8/engine',
                #numpy_include_dir(),
                #],
            #extra_objects=[
                #"gnugo-3.8/engine/libboard.a",
                #"gnugo-3.8/engine/libengine.a",
                #"gnugo-3.8/sgf/libsgf.a",
                #"gnugo-3.8/utils/libutils.a",
                #"gnugo-3.8/patterns/libpatterns.a",
                #],
            #library_dirs=['gnugo-3.8/engine', 'gnugo-3.8/sgf', 'gnugo-3.8/utils'],
            #libraries = ["m", "curses"],
            #),
        #distutils.extension.Extension(
            #"go_loops",
            #["go_loops.pyx"],
            #libraries = ["m", "curses"],
            #),
        distutils.extension.Extension(
            "fuego",
            ["fuego.pyx"],
            language="c++",
            include_dirs = [
                'fuego/go',
                'fuego/gouct',
                'fuego/smartgame',
                'fuego/simpleplayers',
                numpy_include_dir(),
                ],
            extra_objects = [
                "fuego/go/libfuego_go.a",
                "fuego/gouct/libfuego_gouct.a",
                "fuego/simpleplayers/libfuego_simpleplayers.a",
                "fuego/smartgame/libfuego_smartgame.a",
                ],
            #library_dirs=['gnugo-3.8/engine', 'gnugo-3.8/sgf', 'gnugo-3.8/utils'],
            libraries = ["m"],
            ),
        ],
    cmdclass = {'build_ext': Cython.Distutils.build_ext},
    )

