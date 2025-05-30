from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """
    This function reads a requirements file and returns a list of packages.
    """
    requirements=[]
    with open (file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n", "") for req in requirements]    
        
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)
    
    return requirements
    
setup(
    name='Cryptocurrency Liquidity Prediction',
    version='0.0.1',
    author='Jay Tamkhane',
    author_email='jaytamkhane161@gmail.com',
    install_requires=get_requirements('requirements.txt'),
    packages=find_packages(),
)