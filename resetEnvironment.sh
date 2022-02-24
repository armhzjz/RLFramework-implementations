#!/usr/bin/bash


## ATTENTION: This script must be ran only after exiting the python virtual environment

if [ "$1" = "-n" ]; then
    NEW_INSTALL="new install"
else
    NEW_INSTALL=""
fi

FOLDERS_TO_DELETE=("build" "dist" "GridWorld.egg-info" ".venv")
# get this script path
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
# move to the script path
cd "${SCRIPTPATH}"
# if needed, delete not needed folders
for folder in ${FOLDERS_TO_DELETE[@]}; do
    if [ -d  "${folder}" ]; then
        rm -fr "${folder}"
    fi
done

if [ ! -z "${NEW_INSTALL}" ]; then
    # create new virtual environmet
    python -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    python setup.py bdist_wheel
    pip install dist/GridWorld-0.1.0-py3-none-any.whl
fi
