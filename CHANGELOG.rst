=========
Changelog
=========

Version 0.3.0
=============

- Changed format of data structure outputted by reformat()
- Updated README.md to reflect this change

Version 0.2.1
=============

- Moved reformat() function into TDVRPTWInstance class
- Fix bug with assertion check in set_matrices() function
- Print hyperparameters at beginning of run() function

Version 0.2.0
=============

- Created TDVRPTWInstance class to wrap input settings, hyperparameter settings, and algorithm call
- Moved all non-source code files out of package directory
- Reimplemented time to index mapping to support uneven window sizes between indices
- Moved benchmark-specific code to its own file, callable by command line arguments