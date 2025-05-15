#!/bin/bash

function my_kernels_help() {
        if [ $# -lt 1 ]; then
                my_kernels_help list
                echo
                my_kernels_help create
                echo
                my_kernels_help remove
                return
        fi

        case $1 in
        "list")
                echo -e "${Yellow}USAGE: my_kernels_list${NoColor}"
                echo "This helper will list all the currently installed jupyter kernels."
                ;;
        "create")
                echo -e "${Yellow}USAGE: my_kernels_create name [display_name]${NoColor}"
                echo "This helper will create a new kernel for the currently activated virtual environment."
                echo " - name:         name of the kernel you want to create"
                echo " - display_name: name of the kernel, as you want it to appear in the menus, etc."
                echo " e.g.: my_kernels_create my_kernel_py3 \"Python3 (my_venv)\""
                ;;
        "remove")
                echo -e "${Yellow}USAGE: my_kernels_remove name${NoColor}"
                echo "This helper will remove a kernel from the installed kernels list."
                echo " - name:         name of the kernel you want to remove"
                echo " e.g.: my_kernels_remoce my_kernel_py3"
                ;;
        esac
}

function my_kernels_list() {
        echo "Available kernels:"
        KL=$(jupyter kernelspec list | tail -n+2 | sed 's/^\s*//g' | sed 's/\s\+/:/g' | tr '\n' '~' | sed 's/~$//');
        for l in $(echo $KL | tr '~' '\n'); do
                name=$(echo $l | cut -d: -f1)
                path=$(echo $l | cut -d: -f2)
                if [[ "$path" =~ /usr/* ]]; then
                        str=" ${Red}%-10s${NoColor}  %s (default)\n"
                else
                        str=" ${Yellow}%-10s${NoColor}  %s\n"
                fi
                printf " $str" "$name" "$path"
        done
	return 0
}

function my_kernels_create() {
        if [ $# -lt 1 ] || [ "$1" == "help" ] || [ "$1" == "--help" ]; then
                my_kernels_help 'create'
                return
        fi
        p3=$(which python3)
        if [ "$p3" == "/usr/bin/python3" ]; then
                echo "You did not activate a virtual environmenti first!" 1>&2
                echo "Please activate a virtual environment for the kernel you wish to install." 1>&2
                echo "See \"my_venvs_help\" or refer to the documentation." 1>&2
                my_kernels_help 'create'
                return
        fi
        venv=$(echo $p3 | sed 's?/bin/?~?g' | cut -d '~' -f 1)
        name=$1
        dname=${2:-$name}
        eval $p3 -m ipykernel install --user --name \"$name\"  --display-name \"$dname\"
	return $?
}

function my_kernels_remove() {
        if [ $# -lt 1 ] || [ "$1" == "help" ] || [ "$1" == "--help" ]; then
                echo "Please provide the name of the kernel you want to remove." 1>&2
                my_kernels_help 'remove'
                return
        fi
        name=$1
        jupyter kernelspec remove $name
	return $?
}

# EOF
