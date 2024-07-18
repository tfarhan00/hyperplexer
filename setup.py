from setuptools import setup, Extension
import platform

extra_compile_args = ['-O3']
extra_link_args = []

if platform.machine() == 'arm64':
    extra_compile_args.extend(['-arch', 'arm64', '-mfpu=neon'])
    extra_link_args.extend(['-arch', 'arm64'])

module = Extension(
    'hyperplexer',
    sources=['matrix_multiplication.c'],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(
    name='hyperplexer',
    version='1.0',
    description='Matrix multiplication module written in C',
    ext_modules=[module],
)