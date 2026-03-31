# BCP-morphology-map
Python code for building morphology map of BCP using SEM-CLIP and RF

prepare_image.py: cropped SEM image and using auto bright threshold converts it to a black and white contrast image

DataAgu.py: Data augmentation of images 

SEM-CLIP2.py: build the SEM-CLIP classification model

SEM_CLIP_test.py: used the generated SEM-CLIP model to test new images

Accuracy_SEM_CLIP: plot confusion matrix  

BO_homopolymer_RFRmap.py: RF map generation with coress sections plot and active learning suggestions 
