#!/bin/bash

# this script is necessary only because CMake doesn't support (easily) setting
# environment variables in add_custom_command().

BIBTEX_COMPILER=$1
SOURCE_PATH=$2
TEX_BASE_NAME=$3

BSTINPUTS=$SOURCE_PATH BIBINPUTS=$SOURCE_PATH $BIBTEX_COMPILER -terse $TEX_BASE_NAME

