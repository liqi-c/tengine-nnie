# 测试说明 
	编译完成之后的测试主要分两步 ：
	1，搭建NFS服务，建立板卡到设备间的共享目录
	2，拷贝编译完成的资料到共享目录，哪些文件请参考test.sh脚本，然后执行测试程序

# NFS搭建说明
nfs服务能实现将本地PC上面的目录挂载到嵌入式目标板卡，挂载之后可实现在嵌入式目标板卡上面运行pc本地的文件
如下提供一个通过PC虚拟机挂载到海思板卡的说明：
1.	修改IP配置，使需要挂载的虚拟机和板卡在同一个网段，实现网络互通 
如下为网络配置
	虚拟机：   192.168.3.121  
	板子：     192.168.3.168   
	pc本机：   192.168.3.200   

2.	虚拟机nfs的安装和配置   
安装nfs服务：    
	sudo apt-get install nfs-kernel-server     
配置：    
	vim /etc/exports      
	添加：/home/qli 192.168.3.168/32(rw,no_root_squash,fsid=0)    
//  192.168.3.168是板卡的IP    
// /home/qli是虚拟机需要共享的目录   
重启生效：   
	sudo service nfs-kernel-server restart      

查看已经共享的目录：  
	qli@bogon:~$ showmount -e   
	Export list for bogon:  
	/home/qli 192.168.3.168/32  

3.	挂载：将PC目录挂在到板卡上面  
	mount -o nolock,wsize=1024,rsize=1024 192.168.3.121:/home/qli /mnt/nfs  
	
192.168.3.121是需要挂载的主机的IP   
/home/qli 	是虚拟机需要挂载到板卡的目录    
/mnt/nfs    是板卡本地目录，挂载之后在该目录下就可以访问虚拟机的目录/home/qli  

4.	传文件到板卡上面运行   
如上挂载完成之后，将文件传到本地虚拟机的/home/qli目录，之后再到板卡的/mnt/nfs目录下就可以看到你传进去的文件，可以在/mnt/nfs目录下直接执行 。  
