#!/usr/bin/env bash
rm fuego.cpp
rm fuego.so
rm go_loops.c
rm go_loops.so
cython go_loops.pyx
cython fuego.pyx
python setup.py build_ext --inplace

