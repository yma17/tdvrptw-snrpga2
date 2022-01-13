=========
Changelog
=========

Version 0.2.0
=============

- Created TDVRPTWInstance class to wrap input settings, hyperparameter settings, and algorithm call
- Moved all non-source code files out of package directory
- Reimplemented time to index mapping to support uneven window sizes between indices
- Moved benchmark-specific code to its own file, callable by command line arguments