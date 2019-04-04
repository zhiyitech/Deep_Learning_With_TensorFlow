第一步，从视频.mp4文件按照固定的帧差生成图片列表*.jpg  
cd videoclip  
sh run.sh  
cd -   

第二步，使用labelImg,对视频图片进行标注（圈出视频中的leonard）  

第三步，对标注生成的xml进行格式转换，配合视频源文件，生成tfrecord格式的训练输入  
cd preproc  
sh run.sh  
cd -  

第四步，将脚本拷贝至tensorflow的research目录下面，修改对应配置，执行训练。训练结束后执行export脚本输出模型文件。  
#Note that the scripts for train process should be run under tensorflow/models/research directory  
cd trainproc  
sh run.sh  
cd -  

第五步，使用模型文件，对另外一段视频进行目标检测和跟踪。  
cd postproc  
sh run.sh  
cd -  
