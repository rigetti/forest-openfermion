############################################################################
#   Copyright 2017 Rigetti Computing, Inc.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
############################################################################

import sys

from setuptools import setup

if sys.version_info < (3, 6):
    raise ImportError('The forestopenfermion library requires Python 3.6 or above.')

with open('README.md', 'r') as f:
    long_description = f.read()

with open('VERSION.txt', 'r') as f:
    __version__ = f.read().strip()

# save the source code in version.py
with open('forestopenfermion/version.py', 'r') as f:
    version_file_source = f.read()

# overwrite version.py in the source distribution
with open('forestopenfermion/version.py', 'w') as f:
    f.write(f'__version__ = \'{__version__}\'\n')

setup(
    name='forestopenfermion',
    version=__version__,
    author='Rigetti Computing',
    author_email='software@rigetti.com',
    description='A plugin allowing OpenFermion to interface with Forest.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='Apache 2',
    install_requires=[
        'scipy < 1.0.0',
        'numpy < 2.0.0',
        'openfermion < 1.0.0',
        'pyquil < 3.0.0',
        'quantum-grove < 2.0.0'
    ],
    packages=['forestopenfermion'],
    python_requires='>=3.6'
)

# restore version.py to its previous state
with open('forestopenfermion/version.py', 'w') as f:
    f.write(version_file_source)
