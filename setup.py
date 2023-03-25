from setuptools import setup
import sys,os

with open('btbabm/description.txt') as f:
    long_description = f.read()

setup(
    name = 'btbabm',
    version = '0.1.0',
    description = 'bTB Agent based model',
    long_description = long_description,
    url='https://github.com/dmnfarrell/btbabm',
    license='GPL v3',
    author = 'Damien Farrell',
    author_email = 'farrell.damien@gmail.com',
    packages = ['btbabm'],
    package_data={'btbabm': ['data/*.*','logo.png',
                  'description.txt']
                 },
    install_requires=['numpy>=1.2',
                      'pandas>=0.24',
                      'matplotlib>=3.0',
                      'mesa',
                      'geopandas',
                      'toytree',
                      'bokeh',
                      'panel'
                      ],
    entry_points = {
        'console_scripts': [
            'btbabm-dashboard=btb.dashboard:main']
            },
    classifiers = ['Operating System :: OS Independent',
            'Programming Language :: Python :: 3.10',
            'Operating System :: MacOS :: MacOS X',
            'Operating System :: POSIX',
            'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Bio-Informatics'],
    keywords = ['bioinformatics','biology','genomics']
)
