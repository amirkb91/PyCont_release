#!/bin/bash

rm -f /home/akb110/.local/lib/python3.8/site-packages/core.egg-link
rm -f /home/akb110/.local/lib/python3.8/site-packages/easy-install.pth
rm -rf core.egg-info


pip install --user -e .
