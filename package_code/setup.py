from setuptools import setup, find_packages

with open("requirements.txt") as file:
    packages = file.readlines()
    requirements = [package.strip() for package in packages]

print(find_packages())

setup(
    name="ml_model",
    version="0.0.0",
    description="Starter code.",
    author="Student",
    packages=find_packages(),
    install_requires=requirements
)
