#将labelimg输出的xml格式标注数据，转换为更容易读取的csv格式
python xml_to_csv.py
#结合csv标注的视频leonard数据和videoclip第一步生成视频帧图片目录的数据，生成模型输入所需的tfrecord文件
python generate_tfrecord.py --csv_input=leonard_label.csv  --output_path=leonard_labels.record
