from setuptools import setup, find_packages

setup(
    name='tune-listener',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'librosa',
        'pyaudio',
        'matplotlib',
    ],
    entry_points={
        'console_scripts': [
            'tune-listener=tune_listener.__main__:main',
        ],
    },
    author='Your Name',
    description='Listens to a melody and triggers a command when it matches a tune',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
    python_requires='>=3.6',
)
