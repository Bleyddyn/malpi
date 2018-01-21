#! /bin/bash

DIRS=`ls`

for adir in ${DIRS}; do
    if [ -d "${adir}" ]
    then
        cd ${adir}
        echo ${adir}
        MVS=`ls *.h264`
        for amv in ${MVS}; do
            echo ${amv}
            bname=`basename ${amv} .h264`
            mp4name=${bname}.mp4
            if [ ! -f "${mp4name}" ]; then
                echo ${mp4name}
                #MP4Box -add ${amv} ${mp4name}
            fi
        done
        cd ..
    fi
done
