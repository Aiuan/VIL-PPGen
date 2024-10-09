import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


if __name__ == '__main__':
    setup(
        name='custom-package',
        version='1.0.0',
        author='aifuyuan',
        author_email='fyai@zju.edu.cn',
        description='Customize python packages',
        packages=find_packages(),
        cmdclass={
            'build_ext': BuildExtension
        }
    )
