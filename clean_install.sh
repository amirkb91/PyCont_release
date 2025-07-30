#!/bin/bash

rm -rf *.egg-info
rm -rf ~/.local/lib/python3.8/site-packages/__editable__*pycont*
rm -rf ~/.local/lib/python3.8/site-packages/pycont_lib-*.dist-info

pip install --user -e .
