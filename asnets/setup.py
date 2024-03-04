import os

from distutils.ccompiler import new_compiler
from distutils.sysconfig import customize_compiler

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools.dist import Distribution

TF_REQUIRE = 'tensorflow>=2.9.0,<3.0.0'


def _build_ext(ext_obj):
    """Build C/C++ implementation of ASNet-specific TF ops."""
    import tensorflow as tf
    compiler = new_compiler(compiler=None,
                            dry_run=ext_obj.dry_run,
                            force=ext_obj.force)
    customize_compiler(compiler)
    compiler.add_include_dir(tf.sysconfig.get_include())
    src = 'asnets/ops/_asnet_ops_impl.cc'
    dest = 'asnets/ops/_asnet_ops_impl.so'
    # Tests still pass with -Ofast, and it gives marginal speed improvement, so
    # I'm leaving it on. Might revisit later. Also revisit -march=native, which
    # will probably make distribution & Dockerisation a pain.
    extra_flags = ['-march=native', '-Ofast']
    objects = compiler.compile(
        [src],
        debug=True,
        extra_preargs=[*extra_flags, *tf.sysconfig.get_compile_flags()])
    compiler.link(compiler.SHARED_LIBRARY,
                  objects,
                  dest,
                  debug=True,
                  extra_postargs=[
                      *extra_flags, '-lstdc++', '-Wl,--no-undefined',
                      *tf.sysconfig.get_link_flags()
                  ])
    # cleanup: remove object files
    for obj in objects:
        os.unlink(obj)


class build_py_and_ext(build_py):
    # no need to handle planners; we install those on deploy
    def run(self):
        _build_ext(self)
        super().run()


class develop_and_planner_setup_and_ext(develop):
    # here we install planners AND build extensions
    def run(self):
        _build_ext(self)
        super().run()
        from asnets.interfaces.fd_interface import try_install_fd
        try_install_fd()
        from asnets.interfaces.ssipp_interface import try_install_ssipp_solver
        try_install_ssipp_solver()


class install_and_planner_setup(install):
    # no need to build extensions b/c that happens in build_py
    def run(self):
        super().run()
        from asnets.interfaces.fd_interface import try_install_fd
        try_install_fd()
        from asnets.interfaces.ssipp_interface import try_install_ssipp_solver
        try_install_ssipp_solver()


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        # create OS-specific wheels (per example TF op project on Github)
        return True


setup(
    name='asnets',
    version='0.0.1',
    packages=['asnets', 'experiments'],
    # putting this in setup_requires should ensure that we can import tf in
    # _build_ext during setup.py execution; putting it in install_requires
    # ensures that we also have tf at run time
    setup_requires=[
        #TF_REQUIRE
    ],
    install_requires=[
        # these are custom deps that can be a bit painful to install; if they
        # fail, try installing again just in case it was a transient error with
        # the compile scripts
        ('ssipp @ git+https://gitlab.com/qxcv/ssipp.git'
         '#sha1=49e922b4cbdf6d7278a34446ae6fc1732efb6fec'),
        # need to have set up ssh keys and ssh-agent for ANU gitlab for this to
        # work, hopefully will make the repo public at some point
        ('mdpsim @ git+https://gitlab.anu.edu.au/u6937003/mdpsim.git'
         '#sha1=64a54d2b99141d265a24fd193a29c16b2e0756f2'),

        # some of these version numbers are probably not going to work
        # better to change down the line than to have a lower version now

        # these are just vanilla PyPI deps & should be easy to install
        'rpyc>=5.3.0,<5.4.0',
        'tqdm>=4.14.0,<5.0',
        'joblib>=1.2.0,<1.3.0',
        'numpy>=1.24.0,<1.25.0',
        'matplotlib>=3.7.1,<3.8.0',
        'Click>=8.0,<9.0',
        'crayons>=0.4.0,<0.5.0',
        'requests>=2.28.0,<3.0.0',
        'setproctitle>=1.3.2,<1.4.0',
        'JPype1>=1.4.1,<1.5.0',
        'enhsp-wrapper>=0.1.3',
        #TF_REQUIRE,

        # ray (which benefits from psutil/setproctitle/boto3) is required for
        # run_experiments (and maybe other things in the future)
        'psutil>=5.9.4,<5.10.0',
        'ray>=2.3.0,<2.4.0',
        'boto3>=1.26.94,<1.27.0',
        'scikit-optimize>=0.9.0,<0.10.0',
        # ray tune needs pandas for some reason
        # 'pandas>=2.0.3,<2.1.0',

        # for the activation visualisation script (which is optional!), we also
        # need graph_tool, which can be installed manually by following the
        # instructions at https://graph-tool.skewed.de/

        'grpcio<=1.48.2',
    ],
    # include_package_data=True, combined with our MANIFEST.in, ensures that
    # .so files are included
    include_package_data=True,
    distclass=BinaryDistribution,
    zip_safe=False,
    cmdclass={
        'develop': develop_and_planner_setup_and_ext,
        'install': install_and_planner_setup,
        'build_py': build_py_and_ext,
    }
)
