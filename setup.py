# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

# pip install wheel           # 빌드 툴
# pip install setuptools     # 패키징 툴
# pip install twine            # 패키지 업로드 툴

packages = list(open('requirements.txt').readlines())
setup(
    name='aipack',
    version='0.0.5',
    author='HEESEUNG KIM',
    author_email='heewin.kim@gmail.com',
    description='AI Package',
    long_description="""AI Package\n
    본 패키지는 AI Package로써 딥러닝관련한 여러가지 유틸기능들을 포함합니다.\n
    자세한 사용법은 Git 페이지를 확인해주세요.
    """,
    license='MIT',
    url='https://github.com/heewinkim/aipack',
    download_url='https://github.com/heewinkim/aipack/archive/master.zip',

    packages=find_packages(),
    install_requires=open('requirements.txt').readlines(),
    package_data={'':['*']},
    python_requires='>=3.6.1',
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
