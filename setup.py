from setuptools import setup, find_packages

setup(
    name='mempyfit',
    version='0.2.3',
    description='Helper functions for model fitting in Python',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Simon Hansul',
    author_email='simonhansul@gmail.com',
    url='https://github.com/simonhansul/mempyfit.git',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'pyabc',
        'tqdm',
        'yml',
        'matplotlib',
        'seaborn',
        'multipledispatch'
        ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
