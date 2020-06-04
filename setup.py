from setuptools import setup, find_packages

# pip install wheel           # 빌드 툴
# pip install setuptools     # 패키징 툴
# pip install twine            # 패키지 업로드 툴

packages = list(open('requirements.txt').readlines())
setup(
    name='aipack',
    version='0.0.1',
    author='HEESEUNG KIM',
    author_email='heewin.kim@gmail.com',
    description='AI Package',
    long_description=open('README.md').read(),
    license='MIT',
    url='https://github.com/heewinkim/aipack',
    download_url='https://github.com/heewinkim/aipack/archive/master.zip',

    packages=find_packages(),
    install_requires=open('requirements.txt').readlines(),
    package_data={'':['*']},
    python_requires='=3.6.1',
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
