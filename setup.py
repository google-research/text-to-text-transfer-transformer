# Copyright 2022 The T5 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Install T5."""

import os
import sys
import setuptools

# To enable importing version.py directly, we add its path to sys.path.
version_path = os.path.join(os.path.dirname(__file__), 't5')
sys.path.append(version_path)
from version import __version__  # pylint: disable=g-import-not-at-top

# Get the long description from the README file.
with open('README.md') as fp:
  _LONG_DESCRIPTION = fp.read()

setuptools.setup(
    name='t5',
    version=__version__,
    description='Text-to-text transfer transformer',
    long_description=_LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author='Google Inc.',
    author_email='no-reply@google.com',
    url='http://github.com/google-research/text-to-text-transfer-transformer',
    license='Apache 2.0',
    packages=setuptools.find_packages(),
    package_data={
        '': ['*.gin'],
    },
    scripts=[],
    install_requires=[
        'absl-py',
        'babel',
        'editdistance',
        'gin-config',
        'mesh-tensorflow[transformer]>=0.1.13',
        'nltk',
        'numpy',
        'pandas',
        'rouge-score>=0.1.2',
        'sacrebleu',
        'scikit-learn',
        'scipy',
        'sentencepiece',
        'seqio-nightly',
        'six>=1.14',  # TODO(adarob): Remove once rouge-score is updated.
        'tfds-nightly',
        'torch',
        'transformers>=2.7.0',
    ],
    extras_require={
        'gcp': [
            'gevent', 'google-api-python-client', 'google-compute-engine',
            'google-cloud-storage', 'oauth2client'
        ],
        'cache-tasks': ['apache-beam'],
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            't5_mesh_transformer = t5.models.mesh_transformer_main:console_entry_point',
            't5_cache_tasks = seqio.scripts.cache_tasks_main:console_entry_point',
            't5_inspect_tasks = seqio.scripts.inspect_tasks_main:console_entry_point',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='text nlp machinelearning',
)
