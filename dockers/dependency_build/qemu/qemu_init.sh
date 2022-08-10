#!/bin/bash
sudo apt-get install -y qemu binfmt-support qemu-user-static qemu-user
ping bitbucket.itg.ti.com -c 1 > /dev/null 2>&1
if [ "$?" -eq "0" ]; then
    docker run --rm --privileged artifactory.itg.ti.com/docker-public-arm-local/qemu-user-static:5.2.0 --reset -p yes
else    
    docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
fi
