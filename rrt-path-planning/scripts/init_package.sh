#! /bin/bash

find . -type f -not -path "./.git/*" -not -name "scripts/init_package.sh" -exec sed -i "s/rrt-move/$(basename "$PWD")/g" {} +