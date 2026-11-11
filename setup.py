import setuptools

def fetch_requirements():
    with open("requirements.txt") as f:
        reqs = f.read().strip().split("\n")
    return reqs

setuptools.setup(
    name = "compute-trace-analysis",
    description = "A python library for creating and analyzing TensorRT, Jtop profiler traces",
    version = "0.0.0",
    url = "",
    python_requires = ">=3.8",
    author = "Ugent",
    license = "GPL-2.0",
    install_requires = fetch_requirements(),
    include_package_data=True,
    package_dir={
        "cta":"cta"
    },
    classifiers=[
        "Programming Language :: Python 3.9",
        "Programming Language :: Python 3.10",
        "Programming Language :: Python 3.11",
        "Programming Language :: Python 3.12",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: Linux"
    ]
)
