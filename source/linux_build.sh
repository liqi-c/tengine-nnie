
cmake -DCMAKE_TOOLCHAIN_FILE=./linux_toolChain.cmake \
-DTENGINE_PATH=../../../pre-built/linux_arm32 \
-DOPENCV_PATH=/home/qli/Hisi3516CV500/NNIE/opencv/install19a \
-DTENGINE_NNIE_PATH=/home/qli/Hisi3516CV500/Tengine1.7.1-Hi3516cv500/tengine-plugin/NNIE/pre-built \
-DNNIE_SDK_PATH=/home/qli/Hisi3516CV500/Hi3516CV500R001C02SPC001/01.software/board/Hi3516CV500_SDK_V2.0.0.1/package/mpp_smp_linux \
..

#-DNNIE_SDK_PATH=/home/qli/Hisi3516CV500/Hi3516CV500R001C02SPC001/01.software/board/Hi3516CV500_SDK_V2.0.0.1/smp/a7_linux/mpp \ 
#/home/qli/Hisi3516CV500/Hi3516CV500R001C02SPC001/01.software/board/Hi3516CV500_SDK_V2.0.0.1/package/mpp_smp_linux
#-DNNIE_SDK_PATH=/home/qli/Hisi3516CV500/NNIE/sdk19a \
