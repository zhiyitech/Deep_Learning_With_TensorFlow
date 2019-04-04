rm -rf /data/home/goosegu/video/models/research/models_nba/export
mkdir /data/home/goosegu/video/models/research/models_nba/export

INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH=/data/home/goosegu/video/models/research/data/nba/faster_rcnn_resnet101_nba.config
TRAINED_CKPT_PREFIX=/data/home/goosegu/video/models/research/models_nba/model.ckpt-656
EXPORT_DIR=/data/home/goosegu/video/models/research/models_nba/export
python object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}
