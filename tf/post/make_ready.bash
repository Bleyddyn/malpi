#! /bin/bash

DIRS=`ls`

for adir in ${DIRS}; do
    if [ -d "${adir}" ]
    then
        cd ${adir}
        echo ${adir}
        python ../drive_desc.py
        echo "----------"
        cd ..
    fi
done
