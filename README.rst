===============
OPUS-DSD plugin
===============

This plugin provides a wrapper for `OPUS-DSD <https://github.com/alncat/opusDSD>`_ software: Deep structural disentanglement for cryo-EM single-particle analysis.

Installation
-------------

You will need to use 3.0+ version of Scipion to be able to run these protocols. To install the plugin, you have two options:

a) Stable version

.. code-block::

   scipion installp -p scipion-em-opusdsd

b) Developer's version

   * download repository

    .. code-block::

        git clone -b devel https://github.com/scipion-em/scipion-em-opusdsd.git

   * install

    .. code-block::

       scipion installp -p /path/to/scipion-em-opusdsd --devel

OPUS-DSD software will be installed automatically with the plugin but you can also use an existing installation by providing *OPUSDSD_ENV_ACTIVATION* (see below).

**Important:** you need to have conda (miniconda3 or anaconda3) pre-installed to use this program.

Configuration variables
-----------------------
*CONDA_ACTIVATION_CMD*: If undefined, it will rely on conda command being in the
PATH (not recommended), which can lead to execution problems mixing scipion
python with conda ones. One example of this could can be seen below but
depending on your conda version and shell you will need something different:
CONDA_ACTIVATION_CMD = eval "$(/extra/miniconda3/bin/conda shell.bash hook)"

*OPUSDSD_ENV_ACTIVATION* (default = conda activate opusdsd-v1.1.0):
Command to activate the opusdsd environment. 

It will be left the latest version of Opus-DSD as default. In case some other version is required for specific conditions, please, contact us.

Verifying
---------
To check the installation, simply run the following Scipion test:

``scipion test opusdsd.tests.test_protocols_opusdsd.TestOpusDsd``

Supported versions
------------------

0.3.2b

v1.1.0

Protocols
----------

* training CV/Multi
* analyze/eval_vol (volume generation)

References
-----------

1. OPUS-DSD: deep structural disentanglement for cryo-EM single-particle analysis. Zhenwei Luo, Fengyun Ni, Qinghua Wang, Jianpeng Ma. https://www.nature.com/articles/s41592-023-02031-6
