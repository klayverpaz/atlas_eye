import io
import os

from setuptools import find_packages, setup


def read(*paths, **kwargs):
    content = ""

    with io.open(
        os.path.join(os.path.dirname(__file__), *paths), encoding=kwargs.get("encoding", "utf8")
    ) as open_file:
        content = open_file.read().strip()

    return content


def read_requirements(path: str) -> list[str]:
    requirements = [
        line.strip() for line in read(path).split("\n") if not line.startswith(('"', "#", "-", "git+"))
    ]

    return requirements

setup(
    name="odin",
    author="Klayver Paz",
    description="A set of useful tools to deal with different project needs.",
    classifiers=[
        'License :: OSI Approved :: MIT License',
    ],
    packages=find_packages(exclude=[".github"]),
    install_requires=read_requirements("odin/requirements.txt"),
)
