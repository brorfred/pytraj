"""Setup file to generate a distribution of pytraj

usage:    python setup.py sdist
          python setup.py install
"""


from distutils.core import setup

setup(name = 'pytraj',
      version = '0.9',
      description = 'Package to work with lagrangian particle trajectories',
      long_description = "README.md",
      #long_description=open('docs/README.rst', 'rt').read()

      author = 'Bror Jonsson',
      author_email = 'brorfred@gmail.com',
      url = 'http://github.com/brorfred/pytraj',
      requires = ["numpy(>=1.5)",
                  "matplotlib(>=1.1.0)",
                  "mpl_toolkits(>=1.0)",
                  "requests(>=1.1.0)",
                  "projmap(>=0.5)",
                  "njord(>=0.5)",
                  ],
      packages = ['pytraj'],
      package_data = {'pytraj': ['projects.cfg']},
     )
