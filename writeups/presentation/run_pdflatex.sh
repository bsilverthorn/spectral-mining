#!/bin/bash

# this script is necessary only because CMake doesn't support (easily) setting
# environment variables in add_custom_command().

PDFLATEX_COMPILER=$1
INPUT_PATH=$2
TEX_FILE_PATH=$3

#TEXINPUTS=$INPUT_PATH:.:$TEXINPUTS $PDFLATEX_COMPILER -interaction=batchmode $TEX_FILE_PATH
TEXINPUTS=$INPUT_PATH:.:$TEXINPUTS $PDFLATEX_COMPILER $TEX_FILE_PATH

