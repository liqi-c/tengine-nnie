#!/bin/sh
#Author: qli
#Descrition: Use to compile Hisi demo with tengine 

### You need to modify direction TENGINE_SO_PATH and NNIW_SDK_DEPENDENCE_PATH by yourself .
TENGINE_SO_PATH=/home/qli/Hisi3516CV500/Tengine1.7.1-Hi3516cv500
NNIE_SDK_DEPENDENCE_PATH=/home/qli/Hisi3516CV500/Hi3516CV500R001C02SPC001/01.software/board/Hi3516CV500_SDK_V2.0.0.1/package/mpp_smp_linux


###Donn't modify the code below
OPENCV_LIB_PATH=${PWD}/opencv/install19a
cd source
if [ -d build ]; then 
	rm ./build/* -rf 
else 
	mkdir build 
fi

cd build
cmake -DCMAKE_TOOLCHAIN_FILE=../linux_toolChain.cmake \
-DTENGINE_PATH=${TENGINE_SO_PATH}/pre-built/linux_arm32 \
-DOPENCV_PATH=${OPENCV_LIB_PATH} \
-DTENGINE_NNIE_PATH=${TENGINE_SO_PATH}/tengine-plugin/NNIE/pre-built \
-DNNIE_SDK_PATH=${NNIE_SDK_DEPENDENCE_PATH} \
..

make -j4
