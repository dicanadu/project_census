from setuptools import setup, find_packages

print(find_packages())

with open("requirements.txt") as file:
    packages = file.readlines()
    requirements = [package.strip() for package in packages]

setup(
    name="ml_model",
    version="0.0.0",
    description="Starter code.",
    author="Student",
    packages=find_packages(),
    install_requires=requirements,
    package_data={'ml_model': ['sample_models/*.pkl']}
    # include_package_data=True
)


if __name__ == "__main__":
    print(find_packages())
