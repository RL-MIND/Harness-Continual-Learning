from pathlib import Path

from setuptools import setup, find_packages


PKG_NAME = "odyssey"
VERSION = "0.1"
EXTRAS = {}
PROJECT_ROOT = Path(__file__).resolve().parent


def _read_file(fname):
    with (PROJECT_ROOT / fname).open(encoding="utf-8") as fp:
        return fp.read()


def _read_install_requires():
    with (PROJECT_ROOT / "requirements.txt").open(encoding="utf-8") as fp:
        return [
            line.strip()
            for line in fp
            if line.strip() and not line.lstrip().startswith("#")
        ]


def _fill_extras(extras):
    if extras:
        extras["all"] = list(set([item for group in extras.values() for item in group]))
    return extras


setup(
    name=PKG_NAME,
    version=VERSION,
    author=f"MineDojo Team",
    url="https://github.com/zju-vipa/Odyssey",
    description="research project",
    # long_description=_read_file("README.md"),
    long_description_content_type="text/markdown",
    keywords=[
        "Open-Ended Learning",
        "Lifelong Learning",
        "Embodied Agents",
        "Large Language Models",
    ],
    license="MIT License",
    packages=find_packages(include=[PKG_NAME, f"{PKG_NAME}.*"]),
    include_package_data=True,
    zip_safe=False,
    install_requires=_read_install_requires(),
    extras_require=_fill_extras(EXTRAS),
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Environment :: Console",
        "Programming Language :: Python :: 3.9",
    ],
)
