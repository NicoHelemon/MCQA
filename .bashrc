#!/bin/bash

# Source bash functions
BASH_FUNCTIONS=~/.bash_functions
for f in $(find $BASH_FUNCTIONS -type f -name '*\.sh'| sort); do source $f; done

BASH_EXTRA=~/.bash_extra
[[ -r $BASH_EXTRA ]] && source $BASH_EXTRA

# EOF
