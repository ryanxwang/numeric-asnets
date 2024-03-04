# ASNet Dev Container

Using VMs are a bit of a pain, so why not dev containers?

## To use

1. Open this repo in VSCode
2. Install all the Microsoft "remote" extensions
3. Open control palette, and select "Dev Containers: Reopen in Container"

## What does this do?

VSCode will create a dev container according to the Dockerfile. This creates an
ubuntu 22.04 container with the packages we need. After the container is built,
it will run `bash post-create.sh`, which will create the virtual environment and
install all the dependencies.

## Known Issues

- The virtualenv is not automatically activated for each new shell. You need to
  do that yourself (for now).
- `pddl-parser` gives the following warning during installation. Maybe we should
  enforce installing a specific version of Python.
> DeprecationWarning: The distutils package is deprecated and slated for removal
> in Python 3.12. Use setuptools or check PEP 632 for potential alternatives
- Pushing to/pulling from gitlab from the dev container does not work.
