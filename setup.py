from skbuild import setup
import os

try:
    with open(os.path.join(os.path.dirname(__file__), 'VERSION'), 'r') as f:
        version = f.read().strip()
except FileNotFoundError:
    version = os.environ.get("PACKAGE_VERSION", "0.1.0")

try:
    with open(os.path.join(os.path.dirname(__file__), 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "Python bindings for tinyllama.cpp"

setup(
    name="tinyllama_cpp",
    version=version,
    author="Jonathan Reich",
    author_email="jonathanreich100@gmail.com",
    description="Python bindings for the tinyllama.cpp inference engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JohnnyTeutonic/tinyllama.cpp",
    license="MIT",

    packages=['tinyllama_cpp'],

    cmake_install_dir='tinyllama_cpp',

    cmake_args=([
        '-DHAS_CUDA:BOOL=ON'
    ] if os.environ.get('TINYLLAMA_CPP_BUILD_CUDA') == '1' else [
        '-DHAS_CUDA:BOOL=OFF'
    ]),

    install_requires=[
        "numpy", 
    ],
    python_requires='>=3.7',

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 