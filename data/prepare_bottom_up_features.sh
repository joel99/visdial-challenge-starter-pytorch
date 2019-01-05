# Script is responsible for downloading all the relevant datasets, and outputting to a single .h5 for train.py

wget -O data/visdial_1.0_train.zip https://www.dropbox.com/s/ix8keeudqrd8hn8/visdial_1.0_train.zip?dl=1
unzip data/visdial_1.0_train.zip
rm data/visdial_1.0_train.zip

wget -O data/visdial_1.0_val.zip https://www.dropbox.com/s/ibs3a0zhw74zisc/visdial_1.0_val.zip?dl=1
unzip data/visdial_1.0_val.zip
rm data/visdial_1.0_val.zip

wget -O data/visdial_1.0_test.zip https://www.dropbox.com/s/o7mucbre2zm7i5n/visdial_1.0_test.zip?dl=1
unzip data/visdial_1.0_test.zip
rm data/visdial_1.0_test.zip

# Dataloader does preprocessing, would rather get raw data from above
# wget -O data/visdial_data.h5 https://computing.ece.vt.edu/~abhshkdz/visdial/data/v1.0/visdial_data_train.h5
# wget -O data/visdial_params.json https://computing.ece.vt.edu/~abhshkdz/visdial/data/v1.0/visdial_params_train.json

# Get features + config 
wget -O data/features_faster_rcnn_x101_train.h5 https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_faster_rcnn_x101_train.h5
wget -O data/features_faster_rcnn_x101_val.h5 https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_faster_rcnn_x101_val.h5
wget -O data/features_faster_rcnn_x101_test.h5 https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_faster_rcnn_x101_test.h5
wget -O data/visdial_1.0_word_counts_train.json
https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/visdial_1.0_word_counts_train.json


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

# Merge and output h5 - new config renders this unnecessary
# python merge_image_feat.py --save-path data_img_faster_rcnn_x101.h5 --train-path features_faster_rcnn_x101_train.h5 --val-path features_faster_rcnn_x101_val.h5 --test-path features_faster_rcnn_x101_test.h5

# To run training, use lf_disc_faster_rcnn_x101_bs32.yml
# python train.py --config-yml lf_disc_faster_rcnn_x101_bs32.yml --validate

# To run eval on validation (20 epochs) - sanity check this
# python evaluate.py --config-yml lf_disc_faster_rcnn_x101_bs32.yml --use-gt true --load-pthpath checkpoints/checkpoint_19.pth

# Run eval on test
# python evaluate.py --config-yml lf_disc_faster_rcnn_x101_bs32.yml --use-gt false --load-pthpath checkpoints/checkpoint_19.pth --eval-json data/visdial_1.0_test.json