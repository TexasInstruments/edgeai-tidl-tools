#! /bin/bash
mkdir dlr
cd dlr
git clone --depth 1 --single-branch -b master  https://git.ti.com/git/jacinto-linux/meta-psdkla.git
git clone --depth 1 --single-branch -b tidl-j7 https://github.com/TexasInstruments/neo-ai-dlr.git
cd neo-ai-dlr
git submodule init
git submodule update --init --recursive
mkdir build
