from setuptools import find_packages, setup

setup(
    name='GridWorld',
    packages=find_packages(include=['GridWorld']),
    version='0.1.0',
    description='GridWorld library for GridWorld environment evaluations',
    author='armhzjz',
    license='MIT',
    install_requires=["numpy"]
)