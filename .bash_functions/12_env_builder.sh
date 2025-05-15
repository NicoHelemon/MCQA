#!/bin/bash

function kbuilder_help() {
	if [ $# -lt 1 ]; then
                kbuilder_help create
                echo
                kbuilder_help remove
                return
        fi
	case $1 in
	"create")
		echo -e "${Yellow}USAGE: kbuilder_create name (requirements_file|lib_name)${NoColor}"
		echo "This command will create a new virtual environment called 'name' and install"
		echo "the specified python library 'lib_name' OR the pythons libraries from the"
		echo "file 'requirements_file'. Then, a new kernel will be installed in this environment."
		echo "e.g.: kbuilder_create p3-tqdm tqdm"
		echo "e.g.: kbuilder_create p3-data-science ds-requirements.txt"
		;;
	"remove")
		echo -e "${Yellow}USAGE: kbuilder_remove name${NoColor}"
		echo "This command will remove the virtual environment and kernel called 'name'."
		echo "e.g.: kbuilder_remove p3-tqdm"
		;;
	esac
}

function kbuilder_create() {
	if [ $# -lt 2 ]; then
		kbuilder_help create
		return 1
	fi
	name=$1
	libs=$2
	if [ -r $2 ]; then
		libs_type='req'
	else
		libs_type='pip'
	fi
	my_venvs_create $name
	rc=$?
	if [ $rc -ne 0 ]; then return $rc; fi
       	my_venvs_activate $name
	rc=$?
	if [ $rc -ne 0 ]; then return $rc; fi
	case $libs_type in
	"pip")
		pip install $libs
		;;
	"req")
		pip install -r $libs
		;;
	esac
	my_kernels_create $name "$name"
	rc=$?
	if [ $rc -ne 0 ]; then return $rc; fi
	deactivate
	echo -e "${Green}Done! Please refresh your browser's tab to refresh the kernels list.${NoColor}"
	return 0
}

function kbuilder_remove() {
	if [ $# -lt 1 ]; then
		kbuilder_help remove
		return 1
	fi
	name=$1
	my_venvs_remove $name
	rc=$?
	if [ $rc -ne 0 ]; then return $rc; fi
	my_kernels_remove $name
	rc=$?
	if [ $rc -ne 0 ]; then return $rc; fi
	echo -e "${Green}Done! Please refresh your browser's tab to refresh the kernels list.${NoColor}"
	return 0

}

# EOF
