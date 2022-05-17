from setuptools import find_packages, setup

setup(
    name="recommenders",
    version="0.1.0",
    url="https://github.com/inseokson/recommendation-systems",
    author="Inseok Son",
    author_email="inseokson92@gmail.com",
    description="Recommendation system implementation in PyTorch",
    packages=find_packages(exclude=["docs", "notebooks", "tests*"]),
    zip_safe=False,
)
