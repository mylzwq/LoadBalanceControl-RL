"""
loadbalanceRL
--------

Optimizing Wireless network using Reinforcement Learning.

"""

import os
from setuptools import setup, find_packages

DIR = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(DIR, 'README.rst')) as handle:
    README = handle.read()

with open(os.path.join(DIR, 'pip_requirements.txt')) as handle:
    INSTALL_REQUIREMENTS = handle.read()

PROJECT = 'loadbalanceRL'

VERSION = 1.0

CLI = """
[console_scripts]
loadbalanceRL=loadbalanceRL.cli.main:cli
"""

setup(
    name=PROJECT,
    version=VERSION,
    author='Ari Saha, Mingyang Liu',

    description='Optimizing wireless network using Reinforcement Learning',
    long_description=README,

    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    platforms=['Linux', 'Darwin'],
    install_requires=INSTALL_REQUIREMENTS,
    classifiers=[
        'Development Status :: 1 - Alpha',
        'Environment :: Web Environment',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Internet :: WWW/HTTP :: Dynamic Content',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    entry_points=CLI,
)
