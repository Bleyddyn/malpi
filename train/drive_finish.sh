#!/bin/bash

if [ ! -f "current.txt" ]; then
    echo "Missing experiment output file: current.txt"
    exit
fi
if [ ! -f "histories.pickle" ]; then
    echo "Missing experiment history file: histories.pickle"
    exit
fi

NAME=`grep "^Name:" current.txt | cut -c7-`
NAME_FILE=`echo ${NAME} | sed -e "s/ /_/g"`

EXP_DIR="experiments/${NAME_FILE}"
if [ -d "${EXP_DIR}" ]; then
    echo "Error: Experiment directory already exists: ${EXP_DIR}"
    exit
fi

mkdir "${EXP_DIR}"
mv current.txt "${EXP_DIR}/${NAME_FILE}.txt"
python ./plot_history.py --name="${NAME}" histories.pickle
mv "${NAME_FILE}.png" "${EXP_DIR}/"
mv "histories.pickle" "${EXP_DIR}/"
