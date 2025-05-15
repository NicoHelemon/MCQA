#!/bin/bash

function my_venvs_help() {
        if [ $# -lt 1 ]; then
                my_venvs_help 'list'
                echo
                my_venvs_help 'create'
                echo
                my_venvs_help 'activate'
                echo
                my_venvs_help 'deactivate'
                echo
                my_venvs_help 'remove'
                return
        fi

        case $1 in
        "list")
                echo -e "${Yellow}USAGE: my_venvs_list [--pip-list]${NoColor}"
                echo "--pip-list (or -p): will list the installed pip packages"
                echo "This helper will list all the available virtual environments."
                ;;
        "create")
                echo -e "${Yellow}USAGE: my_venvs_create name${NoColor}"
                echo "This helper will create a new virtual environment."
                echo " - name:         name of the virtual environment"
                echo " e.g.: my_venvs_create data_science"
                ;;
        "activate")
                echo -e "${Yellow}USAGE: my_venvs_activate name${NoColor}"
                echo "This helper will activate an existing virtual environment."
                echo "When a virtual environment is activated, you can install more packages"
                echo "and add kernels (type \"my_kernels_help\" to learn more."
                echo " - name:         name of the virtual environment"
                echo " e.g.: my_venvs_activate data_science"
                ;;
        "deactivate")
                echo -e "${Yellow}USAGE: my_venvs_deactivate name${NoColor}"
                echo "This helper will deactivate the currently activated virtual environment."
                echo " - name:         name of the virtual environment"
                echo " e.g.: my_venvs_deactivate data_science"
                ;;
         "remove")
                echo -e "${Yellow}USAGE: my_venvs_remove name${NoColor}"
                echo "This helper will remove (delete) a virtual environment."
                echo " - name:         name of the virtual environment you want to remove"
                echo " e.g.: my_venvs_remove data_science"
                ;;
        esac

}

function my_venvs_list() {
        if [ "$1" == "help" ] || [ "$1" == "--help" ]; then
                my_venvs_help 'list'
                return 0
        fi
        LIST=0
        if [ $# -gt 0 ]; then
                case $1 in
                "--pip-list"|"-p")
                        LIST=1
                        shift
                        ;;
                esac
        fi
        local venv=""
        if [ $LIST -eq 1 ]; then
                p3=$(which python3)
                if [ "$p3" != "/bin/python3" ]; then
			# Save the currently active virtual env
                        venv=$(echo $p3 | rev | cut -d '/' -f 2- | rev)
			# We need to deactivate the current virtual env.
                        deactivate
                fi
        fi
        echo "Available virtual environments:"
        t=0
        for v in $(find $MY_VENVS -mindepth 1 -maxdepth 1 -type d); do
                t=$((t+1))
                echo -e "  "${Yellow}$(basename $v)${NoColor}" ("$(du -sh $v | cut -f 1)")"
                if [ $LIST -eq 1 ]; then
                        source $v/bin/activate
			python3 -m pip list --local --format=columns
			deactivate
                fi
        done
        echo "$t virtual environment(s) found."
        if [ -n "$venv" ]; then
                if [ -r $venv/activate ]; then
                        source $venv/activate
                fi
        fi
        if [ $LIST -eq 0 ]; then
                echo "Tip: use \"my_venvs_list --pip-list\" to list installed pip packages."
        fi
        return 0
}

function my_venvs_create() {
        if [ $# -lt 1 ] || [ "$1" == "help" ] || [ "$1" == "--help" ]; then
                my_venvs_help 'create'
                return 1
        fi
        if [ $# -lt 1 ]; then
                echo "Please provide a name for the virtual environment!" 1>&2
                my_venvs_help 'create'
                return 1
        fi
        VENV=$1
        if [ -d $MY_VENVS/$VENV ]; then
                echo "Environment '$MY_VENVS/$VENV' already exists."
                return 2
        fi
        # Create venv
	#CMD="/usr/bin/python3 -m venv $MY_VENVS/$VENV --system-site-packages"
        VENV_BIN=$(which virtualenv)
        CMD="$VENV_BIN --system-site-packages --activators bash,python --download $MY_VENVS/$VENV"
        echo "FYI, I am running: $CMD"
        echo "Please wait, this will take some time."
        eval $CMD
        return $?
}

function my_venvs_activate() {
        if [ $# -lt 1 ] || [ "$1" == "help" ] || [ "$1" == "--help" ]; then
                my_venvs_help 'activate'
                return 1
        fi
        if [ $# -lt 1 ]; then
                echo "Please provide a name for the virtual environment!" 1>&2
                my_venvs_help 'activate'
                return 1
        fi
        VENV=$1
        if [ -r $MY_VENVS/$VENV/bin/activate ]; then
                source $MY_VENVS/$VENV/bin/activate
                return 0
        else
                echo -e "${Red}Virtual environment not found${NoColor}: $VENV" 1>&2
                return 1
        fi
}

function my_venvs_deactivate() {
        if [ "$1" == "help" ] || [ "$1" == "--help" ]; then
                my_venvs_help 'activate'
                return 0
        fi
        deactivate > /dev/null 2>&1
        return 0
}

function my_venvs_remove() {
        if [ $# -lt 1 ] || [ "$1" == "help" ] || [ "$1" == "--help" ]; then
                my_venvs_help 'remove'
                return 1
        fi
        if [ $# -lt 1 ]; then
                echo "Please provide a name for the virtual environment!" 1>&2
                my_venvs_help 'remove'
                return 1
        fi
        VENV=$1
        if [ -d $MY_VENVS/$VENV ]; then
                echo "I am about to remove \"$MY_VENVS/$VENV\"."
                echo -ne "Type \"${Red}yes${NoColor}\" to proceed: "
                read a
                if [ "$a" == "yes" ]; then
                        rm -rf $MY_VENVS/$VENV
                        return 0
                else
                        echo "Aborting."
                        return 1
                fi
        else
                echo -e "${Red}Virtual environment not found${NoColor}: $VENV" 1>&2
                return 2
        fi
}

# EOF
