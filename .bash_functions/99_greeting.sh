#!/bin/bash

# Prompt and welcome message
PS1="\[\e[1;31m\]${UNIX_USERNAME}\[\e[m\]@\[\e[1;37m\]\h\[\e[m\]:\[\e[1;37m\]\w\[\e[m\] \[\e[1;31m\]\\$\[\e[m\] "
clear
echo -e "${Red}Welcome to ${White}Jupyter${Red}!${NoColor}"

# EOF
