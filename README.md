# tengine-nnie
Tengine example for run nnie devices。
功能说明： 
	该工程适用于海思3516CV500,3516DV300,3519AV100 三个硬件平台下用tengine进行NN推理。
	目前支持的网络模型有 ：FasterrrcnnAlexnet, cnn ,ssd, yolov1, yolov2, yolov3 

该库有两个依赖：
	1，依赖海思的编译链，如步骤1所示，如果没有安装过需要按照步骤1安装。
	2，依赖tengine的版本库 ，按照步骤2下载对应版本

# Step 1 :  安装海思的编译器 
  解压海思资料：arm-himix200-linux.tgz  
  cd arm-himix200-linux
  chmod  arm-himix200-linux.install 
  sourch ./arm-himix200-linux.install
  ./arm-himix200-linux.install 

  安装完成之后输入arm-himix200-linux- 会提示有哪些编译工具
  $ arm-himix200-linux-
  arm-himix200-linux-addr2line   arm-himix200-linux-g++         arm-himix200-linux-gcov-tool   arm-himix200-linux-ranlib
  arm-himix200-linux-ar          arm-himix200-linux-gcc         arm-himix200-linux-gprof       arm-himix200-linux-readelf
  arm-himix200-linux-as   
  
# Step 2 : 下载Tengine的对应的版本库 
  更新地址： http://www.openailab.com/info.php?class_id=102101
  
# Step 3 : 下载安装海思的依赖库 
  下载链接：https://pan.baidu.com/s/13cEi_omGo-DkrnKdqwjGew    提取码：tukl 
  unrar Hi3516CV500R001C02SPC001.rar 
  cd Hi3516CV500R001C02SPC001/01.software/board  
  tar xzvf  Hi3516CV500_SDK_V2.0.0.2.tgz
  bash  sdk.unpack 
  此时会在当前目录生成：smp\a7_linux\mpp ，该目录路径下面操作需要配置

# Step 4 ： 修改编译脚本 
linux_build.sh 修改如下两个目录 ：
#### TENGINE_SO_PATH 这个路径来自于步骤2下载的目录主路径 ：
    TENGINE_SO_PATH=/home/qli/Hisi3516CV500/Tengine1.7.1-Hi3516cv500
#### NNIE_SDK_DEPENDENCE_PATH 来自于步骤3里面安装的路径     
    NNIE_SDK_DEPENDENCE_PATH=/home/qli/Hisi3516CV500/Hi3516CV500R001C02SPC001/01.software/board/Hi3516CV500_SDK_V2.0.0.1/package/mpp_smp_linux

# Step5 ：编译 
    直接执行linux_build.sh 脚本即可

# Step5 ：执行 
    请参照Test目录下的脚本实现

