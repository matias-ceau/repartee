from setuptools import setup, find_packages

setup(
    name='repartee',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'requests',
        'PyYAML',
        'prompt_toolkit',
    ],
    entry_points={
        'console_scripts': [
            'repartee=repartee.cli:main',
        ],
    },
)