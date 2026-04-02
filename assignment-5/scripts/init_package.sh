#! /bin/bash

find . -type f -exec sed -i "s/robot_localization_pf_student/$(basename "$PWD")/g" {} +
