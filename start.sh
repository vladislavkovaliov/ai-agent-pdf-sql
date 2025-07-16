#!/bin/bash

files=(
  "1.pdf"
  "2.pdf"
  "3.pdf"
  "4.pdf"
  "5.pdf"
  "6.pdf"
  "7.pdf"
)

DYLD_LIBRARY_PATH=/opt/homebrew/lib python3 main.py "${files[@]}"
