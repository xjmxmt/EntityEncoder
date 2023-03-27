from setuptools import setup, find_packages

setup(
    name = 'entityencoder',
    version = '0.1.0',
    keywords='entity embedding',
    description = 'a library for convert discrete features to continuous features',
    license = 'MIT License',
    url = 'https://github.com/xjmxmt/EntityEncoder',
    author = 'Jiaming Xu',
    author_email = 'riverfalling1001@gmail.com',
    packages = find_packages(),
    include_package_data = True,
    platforms = 'any',
    install_requires = [],
)