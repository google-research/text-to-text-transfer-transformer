# Copyright 2019 The T5 Authors.
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

import setuptools

setuptools.setup(
    name='t5',
    version='0.1.4',
    description='Text-to-text transfer transformer',
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
        'allennlp',
        'babel',
        'future',
        'gin-config',
        'mesh-tensorflow[transformer]',
        'nltk',
        'numpy',
        'pandas',
        'rouge-score',
        'sacrebleu',
        'scikit-learn',
        'scipy',
        'sentencepiece',
        'six',
        'tensorflow-datasets>=1.3.0',
        'tensorflow-text==1.15.0rc0',
    ],
    extras_require={
        'tensorflow': ['tensorflow==1.15'],
        'gcp': ['gevent', 'google-api-python-client', 'google-compute-engine',
                'google-cloud-storage', 'oauth2client'],
    },
    entry_points={
        'console_scripts': [
            't5_mesh_transformer = '
            't5.models.mesh_transformer_main:console_entry_point',
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
