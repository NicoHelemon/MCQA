#!/bin/bash

if [ ! -t 1 ]; then return; fi
if [ ! -t 2 ]; then return; fi

# Define colors
Black="\e[0;30m"
DarkGray="\e[1;30m"
DarkBlue="\e[0;34m"
Blue="\e[1;34m"
DarkGreen="\e[0;32m"
Green="\e[1;32m"
Cyan="\e[0;36m"
LightCyan="\e[1;36m"
DarkRed="\e[0;31m"
Red="\e[1;31m"
Purple="\e[0;35m"
Pink="\e[1;35m"
Brown="\e[0;33m"
Yellow="\e[1;33m"
Gray="\e[0;37m"
White="\e[1;37m"
NoColor="\e[00m"

function colors() {
        echo -e ${Black}Black$NoColor
        echo -e ${DarkGray}DarkGray$NoColor
        echo -e ${DarkBlue}DarkBlue$NoColor
        echo -e ${Blue}Blue$NoColor
        echo -e ${DarkGreen}DarkGreen$NoColor
        echo -e ${Green}Green$NoColor
        echo -e ${Cyan}Cyan$NoColor
        echo -e ${LightCyan}LightCyan$NoColor
        echo -e ${DarkRed}DarkRed$NoColor
        echo -e ${Red}Red$NoColor
        echo -e ${Purple}Purple$NoColor
        echo -e ${Pink}Pink$NoColor
        echo -e ${Brown}Brown$NoColor
        echo -e ${Yellow}Yellow$NoColor
        echo -e ${Gray}Gray$NoColor
        echo -e ${White}White$NoColor
        echo -e ${NoColor}NoColor$NoColor
}

# EOF
