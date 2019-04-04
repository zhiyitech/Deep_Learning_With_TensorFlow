cd videoclip  
sh run.sh  
cd -  

cd preproc  
sh run.sh  
cd -  

#Note that the scripts for train process should be run under tensorflow/models/research directory  
cd trainproc  
sh run.sh  
cd -  

cd postproc  
sh run.sh  
cd -  
