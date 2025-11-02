from setuptools import setup

setup(
    name='fom_placeholder',
    version='0.1.0',    
    description='Figure of Merit for Photovoltaics',
    url='https://github.com/SMTG-Bham/PV-FoM',
    author='Philippa U Cox, Peter P Russell',
    author_email='puc369@student.bham.ac.uk, ppr466@student.bham.ac.uk',
    license='MIT',
    packages=['placeholder'],
    install_requires=['numpy',
                      'scipy',
                      'matplotlib>=3.9.4',
                      'pymatgen>=2024.8.9',
                      'mpmath',
                      'pathlib',
                      'typing_extensions',
                      'monty'                     
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.9',
    ],
)