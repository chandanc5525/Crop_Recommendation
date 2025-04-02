from setuptools import setup, find_packages


with open('requirements.txt', 'r') as f:
    install_requires = f.read().splitlines()
    
    
    setup(
        name='Crop Recommendation Model',
        version='1.0.0',
        description='Tnis is a recommendation model for crops based on weather and soil conditions.',
        author='Chandan Chaudhari',
        author_email= 'chaudhari.chandan22@gmail.com',)