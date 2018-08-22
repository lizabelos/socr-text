from distutils.core import setup
from distutils.extension import Extension

from Cython.Build import cythonize
from Cython.Distutils import build_ext

extensions = [
    Extension("coder.ctc_decoder", ["coder/ctc_decoder.pyx"]),
    Extension("rating.word_error_rate", ["rating/word_error_rate.pyx"]),
    Extension("coder.language.beam", ["coder/language/beam.pyx"]),
    Extension("coder.language.prefix_tree", ["coder/language/prefix_tree.pyx"]),
    Extension("coder.language.language_model", ["coder/language/language_model.pyx"]),
    Extension("coder.language.word_beam_search", ["coder/language/word_beam_search.pyx"]),
    Extension("loss.ctc", ["loss/ctc.pyx"]),
    Extension("modules.ctc", ["modules/ctc.pyx"]),
]

setup(
    cmdclass={
        'build_ext': build_ext
    },
    ext_modules=cythonize(extensions),
)
