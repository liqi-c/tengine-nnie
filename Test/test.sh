#!/bin/bash
# 该脚本在主机环境下运行，需要配置TENGINE_ROOT和NNIE_SDK_PATH的目录，执行目录为挂载的共享目录
#  执行脚本的目的是为了将依赖文件拷贝到共享目录，然后后面再板卡侧能顺利执行测试程序

#TENGINE_ROOT=/home/qli/Hisi3516CV500/Tengine1.7.1-Hi3516cv500/
TENGINE_ROOT=/home/qli/TE-BU-S000-Hi3516DV300-1.8.0-R20190925
NNIE_SDK_PATH=/home/qli/Hisi3516CV500/Hi3516CV500R001C02SPC001/01.software/board/Hi3516CV500_SDK_V2.0.0.1/smp/a7_linux/mpp

if [ "$1" = "LIB" ]; then 
	cp ${NNIE_SDK_PATH}/lib/libnnie.so ./
	
	cp ${TENGINE_ROOT}/pre-built/linux_arm32/lib/libhclcpu.so   ./
	cp ${TENGINE_ROOT}/pre-built/linux_arm32/lib/libtengine.so  ./
	cp ${TENGINE_ROOT}/tengine-plugin/NNIE/pre-built/lib/*      ./  -rf
	
	cp ../source/build/sample_nnie_plugin   ./
	cp ../source/build/test_nnie_all   ./

	cp ../opencv/install19a/lib/libopencv_core.so.2.4     ./
	cp ../opencv/install19a/lib/libopencv_highgui.so.2.4  ./
	cp ../opencv/install19a/lib/libopencv_imgproc.so.2.4  ./
	
	cp ${NNIE_SDK_PATH}/sample/svp/nnie/data  ./ -rf
	cp ${NNIE_SDK_PATH}/lib/* ./  -rf 

	cp ./inst_yolov3_cycle.wk ./data/nnie_model/detection/inst_yolov3_cycle.wk
	cp ./dog_bike_car_416x416.bgr ./data/nnie_image/rgb_planar/dog_bike_car_416x416.bgr
fi

if [ "$1" = "RUN" ]; then 
	echo " Run Test "
	
	### 这里面提供了海思的一些库，如果你板卡当前目录下无此目录，则需要拷贝
	##    Hi3516CV500R001C02SPC001/01.software/board/Hi3516CV500_SDK_V2.0.0.1/smp/a7_linux/mpp/lib/ 
	##  cp ${NNIE_SDK_PATH}/lib/* ./  -rf 
	##  目录到板卡对应的执行目录
	export LD_LIBRARY_PATH=./   
	
	## openmp优化需要 该库是编译器中的,SDK里面没有
	cp /opt/hisi-linux/x86-arm/arm-himix200-linux/arm-linux-gnueabi/lib/libgomp.so* ./

	./sample_nnie_plugin
	./test_nnie_all -m 2
fi

#海思自己的测试程序  
#Hi3516CV500R001C02SPC001\01.software\board\Hi3516CV500_SDK_V2.0.0.1\smp\a7_linux\mpp\sample\svp\nnie

  
# 依赖文件说明 
## void TEST_NNIE_FasterRcnn()
#{
#    const char *image_file = "./data/nnie_image/rgb_planar/single_person_1240x375.bgr";
#    const char *model_file = "./data/nnie_model/detection/inst_alexnet_frcnn_cycle.wk";
#    const char *cpuConfigFile = "./inst_alexnet_frcnn_cycle_cpu.cfg";
##
##void TEST_NNIE_Cnn()
#{
#    const char *image_file = "./data/nnie_image/y/0_28x28.y";
#    const char *model_file = "./data/nnie_model/classification/inst_mnist_cycle.wk";
##void TEST_NNIE_Ssd()
#{
#    // const char *image_file = "./data/nnie_image/rgb_planar/dog_bike_car_300x300.bgr";
#    const char *image_file_org = "./data/nnie_image/rgb_planar/dog_bike_car.jpg";
#    const char *model_file = "./data/nnie_model/detection/inst_ssd_cycle.wk";
#
#void TEST_NNIE_Yolov1()
#{
#    const char *image_file = "./data/nnie_image/rgb_planar/dog_bike_car_448x448.bgr";
#    const char *model_file = "./data/nnie_model/detection/inst_yolov1_cycle.wk";	
#	
#void TEST_NNIE_Yolov2()
#    const char *image_file = "./data/nnie_image/rgb_planar/street_cars_416x416.bgr";
#    const char *image_file_org = "./data/nnie_image/rgb_planar/street_cars.png";
#    const char *model_file = "./data/nnie_model/detection/inst_yolov2_cycle.wk";	
#
#void TEST_NNIE_Yolov3()
#    const char *image_file = "./data/nnie_image/rgb_planar/dog_bike_car_416x416.bgr";
#    const char *image_file_org = "./data/nnie_image/rgb_planar/dog_bike_car.jpg";
#    const char *model_file = "./data/nnie_model/detection/inst_yolov3_cycle.wk"; 
#
#void TEST_NNIE_Lstm()
#    const char *image_file[3] = {"./data/nnie_image/vector/Seq.SEQ_S32",
#                                 "./data/nnie_image/vector/Vec1.VEC_S32",
#                                 "./data/nnie_image/vector/Vec2.VEC_S32"};
#    const char *model_file = "./data/nnie_model/recurrent/lstm_3_3.wk";
#void TEST_NNIE_Pvanet()
#    const char *image_file = "./data/nnie_image/rgb_planar/horse_dog_car_person_224x224.bgr";
#    const char *model_file = "./data/nnie_model/detection/inst_fasterrcnn_pvanet_inst.wk";
#