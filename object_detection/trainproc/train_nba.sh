#wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
#wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
#tar -xvf images.tar.gz
#tar -xvf annotations.tar.gz

#python object_detection/dataset_tools/create_pet_tf_record.py \
#    --label_map_path=object_detection/data/pet_label_map.pbtxt \
#    --data_dir=`pwd` \
#    --output_dir=`pwd`

#wget http://storage.googleapis.com/download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz
#tar -xvf faster_rcnn_resnet101_coco_11_06_2017.tar.gz
#mv faster_rcnn_resnet101_coco_11_06_2017/model.ckpt.* data/

# From the tensorflow/models/research/ directory
PIPELINE_CONFIG_PATH="/data/home/goosegu/video/models/research/data/nba/faster_rcnn_resnet101_nba.config"  #{path to pipeline config file}
MODEL_DIR=models_nba
NUM_TRAIN_STEPS=1000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr
