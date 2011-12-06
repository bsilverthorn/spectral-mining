#!/usr/bin/env bash
rm gnugo_engine.c
rm gnugo_engine.so
rm go_loops.c
rm go_loops.so
rm fuego.c
rm fuego.so
cython gnugo_engine.pyx
cython go_loops.pyx
cython fuego.pyx
python setup.py build_ext --inplace
