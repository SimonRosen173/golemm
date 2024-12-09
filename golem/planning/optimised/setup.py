from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# module = Extension('dijkstra', sources=['dijkstra.pyx'], include_dirs=[numpy.get_include()])
#
# setup(
#     name="build_ext",
#     # ext_modules=cythonize(["*.pyx"], annotate=True),
#     ext_modules=[module]
# )

setup(
    name="build_ext",
    ext_modules=cythonize(["*.pyx"], annotate=True),
    include_dirs=[numpy.get_include()]
)
