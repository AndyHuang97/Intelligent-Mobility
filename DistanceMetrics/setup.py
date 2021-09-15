from setuptools import setup
from setuptools.extension import Extension
from distutils import sysconfig
from Cython.Distutils import build_ext
import os
import numpy
from Cython.Build import cythonize

class NoSuffixBuilder(build_ext):
    def get_ext_filename(self, ext_name):
        filename = super().get_ext_filename(ext_name)
        suffix = sysconfig.get_config_var('EXT_SUFFIX')
        ext = os.path.splitext(filename)[1]
        return filename.replace(suffix, "") + ext

extensions = [Extension(
        name="heom_c1",
        sources=["heom_c.pyx"],
        include_dirs=[numpy.get_include()],
        ),
        Extension(
        name="euclidean",
        sources=["euclidean.pyx"],
        include_dirs=[numpy.get_include()],
        )
    ]

setup(
    ext_modules = cythonize(extensions),
    cmdclass={"build_ext": NoSuffixBuilder},
)
