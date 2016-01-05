from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
setup(cmdclass={'build_ext':build_ext}, ext_modules = [Extension("flutils", ["flutils.pyx"])])
setup(cmdclass={'build_ext':build_ext}, ext_modules = [Extension("fieldline", ["fieldline.pyx"])])
setup(cmdclass={'build_ext':build_ext}, ext_modules = [Extension("fields", ["fields.pyx"])])
