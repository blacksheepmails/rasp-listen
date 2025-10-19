from setuptools import setup, find_packages

setup(
    name='tune-listener',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'librosa',
        'pyaudio', # Note: pyaudio can be tricky to install on some systems
        'scipy',
    ],
    entry_points={
        'console_scripts': [
            'tune-listener = tune_listener.__main__:main',
        ],
    },
    description='Listen for a short tune and execute a shell command.',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)