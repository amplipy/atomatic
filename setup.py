from setuptools import setup, find_packages

setup(
    name='atomatic',
    version='0.1',
    author='Petro Maksymovych',
    author_email='pmaksym@clemson.edu',
    maintainer='Petro Maksymovych',
    maintainer_email='pmaksym@clemson.edu',
    packages=find_packages(),
    description='(semi)automatic analysis of molecular, atomic, particle images',
    long_description=open('README.md').read(),
    install_requires=['tqdm','python-pptx','pyperclip','matplotlib-scalebar','xarray'],
    url='https://github.com/amplifilo/nanonisxarray',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
