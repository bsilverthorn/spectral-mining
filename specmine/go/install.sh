#!/usr/bin/env bash
rm fuego.cpp
rm fuego.so
rm go_loops.c
rm go_loops.so
python setup.py build_ext --inplace

