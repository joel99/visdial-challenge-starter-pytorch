# Script is responsible for downloading all the relevant datasets, and outputting to a single .h5 for train.py

# Assumes train val test splits (as opposed to train_val/split)

# Misc visdial preprocessed data
wget -O data/visdial_data.h5 https://computing.ece.vt.edu/~abhshkdz/visdial/data/v1.0/visdial_data_train.h5
wget -O data/visdial_params.json https://computing.ece.vt.edu/~abhshkdz/visdial/data/v1.0/visdial_params_train.json

# Get image features
wget -O data/features_faster_rcnn_x101_train.h5 https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_faster_rcnn_x101_train.h5
wget -O data/features_faster_rcnn_x101_val.h5 https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_faster_rcnn_x101_val.h5
wget -O data/features_faster_rcnn_x101_test.h5 https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_faster_rcnn_x101_test.h5

# Below is discarded for generating/combining actual features
# # Train Image Features TrainVal (discard val)
# wget -P data https://imagecaption.blob.core.windows.net/imagecaption/trainval_36.zip
# unzip data/trainval_36.zip -d data
# rm data/trainval_36.zip

# # Convert train image features into h5
# python detection_features_converter.py

# # Validation Images
# wget -O data/val_images.zip https://www.dropbox.com/s/twmtutniktom7tu/VisualDialog_val2018.zip?dl=1
# unzip data/val_images.zip
# rm data/val_images.zip

# # Test Images
# wget -O data/test_images.zip https://www.dropbox.com/s/mwlrg31hx0430mt/VisualDialog_test2018.zip?dl=1
# unzip data/test_images.zip
# rm data/test_images.zip

# # Extract validation and test image features into h5 TODO: needs detectron config
# python extract_bottomup.py --image-root data/val_images --split val --save-path data_img_val.h5
# python extract_bottomup.py --image-root data/test_images --split test --save-path data_img_test.h5

# Merge and output h5
python merge_image_feat.py --save-path data_img_faster_rcnn_x101.h5 --train-path features_faster_rcnn_x101_train.h5 --val-path features_faster_rcnn_x101_val.h5 --test-path features_faster_rcnn_x101_test.h5