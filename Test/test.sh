#!/bin/bash

TENGINE_ROOT=/home/qli/Hisi3516CV500/Tengine1.7.1-Hi3516cv500/
NNIE_SDK_PATH=/home/qli/Hisi3516CV500/Hi3516CV500R001C02SPC001/01.software/board/Hi3516CV500_SDK_V2.0.0.1/smp/a7_linux/mpp

if [ "$1" = "LIB" ]; then 
	cp ${NNIE_SDK_PATH}/lib/libnnie.so ./
	
	cp ${TENGINE_ROOT}/pre-built/linux_arm32/lib/libhclcpu.so   ./
	cp ${TENGINE_ROOT}/pre-built/linux_arm32/lib/libtengine.so  ./
	cp ${TENGINE_ROOT}/tengine-plugin/NNIE/pre-built/lib/*      ./  -rf
	
	cp ${TENGINE_ROOT}/tengine-plugin/NNIE/source/build/sample_nnie_plugin   ./
	cp ${TENGINE_ROOT}/tengine-plugin/NNIE/source/test_yu.jpg.rgb ./
	cp ${TENGINE_ROOT}/tengine-plugin/NNIE/source/inst_mnist_cycle.wk ./
	cp ${TENGINE_ROOT}/tengine-plugin/NNIE/source/build/test_nnie_all   ./

	cp /home/qli/Hisi3516CV500/NNIE/sdk19a/sample/svp/nnie/data  ./ -rf
	cp /home/qli/Hisi3516CV500//NNIE/nnieplugin/test/inst_alexnet_frcnn_cycle_cpu.cfg  ./ -rf
fi

if [ "$1" = "RUN" ]; then 
	echo " Run Test "
	export LD_LIBRARY_PATH=/mnt/ko/libs
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