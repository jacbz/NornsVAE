from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but it might need
# fine tuning.
build_options = {"build_exe": "build",
                 'packages': ["tf_slim.layers", "scipy.integrate", "scipy.optimize", "keras"],
                 "include_files": ["assets/"],
                 'excludes': []}

base = 'Console'

executables = [
    Executable('server.py', base=base)
]

setup(name='nornsvae_server',
      version='1.0',
      description='',
      options={'build_exe': build_options},
      executables=executables)
