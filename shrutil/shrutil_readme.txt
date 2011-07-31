Generic utilities for NVIDIA GPU Computing SDK
==============================================

SHRUTIL is a generic utility library designed for use in the NVIDIA GPU Computing SDK.

It provides functions for:
- logging to console or file
- reading and writing to and from image or data files
- comparing arrays or files with epsilon and threshold
- parsing command line arguments

SHRUTIL is not part of CUDA
===========================

Note that SHRUTIL is not part of the CUDA Toolkit and is not supported by NVIDIA.
It exists only for the convenience of writing concise and platform-independent
example code. 

Library Functions
=================

Most of the functions should be self explanatory. The function parameters are
documented in the "shrUtil.h" file.

Macros
======

shrCheckErrorEX
- Full error handling macro with Cleanup() callback (if supplied).

shrCheckError
- Short version without Cleanup() callback pointer.

shrEXIT
- Standardized Exit Macro for leaving main().

ARGCHECK
- Simple argument checker macro.

STDERROR
- Define for user-customized error handling.