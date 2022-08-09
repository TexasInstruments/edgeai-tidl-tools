#!/bin/bash
# docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
sudo apt-get install -y qemu binfmt-support qemu-user-static qemu-user
docker run --rm --privileged artifactory.itg.ti.com/docker-public-arm-local/qemu-user-static:5.2.0 --reset -p yes