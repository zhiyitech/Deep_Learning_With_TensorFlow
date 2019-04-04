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
