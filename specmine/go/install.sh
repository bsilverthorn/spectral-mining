#!/usr/bin/env bash
rm fuego.cpp
rm fuego.so
cython fuego.pyx
python setup.py build_ext --inplace

