# cs231-capsule-yolo-traffic-sign-detection
Course project for CS231, modified YOLO architecture by adding capsule network for classification

put GTSRB and GTSDB data into data folder then run

`python build_data.py`

train model, modify the corresponding params.json file inside the experiment/model_name/ folder.

`python main.py --model <'model_name'>`

run tensorboardx:

`tensorboard --logdir=<'path_to_log'>`







