#!/bin/bash

# this script is necessary only because CMake doesn't support (easily) setting
# environment variables in add_custom_command().

INPUT_PATH=$1
TEX_FILE_PATH=$2

#TEXINPUTS=$INPUT_PATH:.:$TEXINPUTS xelatex -interaction=batchmode $TEX_FILE_PATH
TEXINPUTS=$INPUT_PATH:.:$TEXINPUTS xelatex $TEX_FILE_PATH

