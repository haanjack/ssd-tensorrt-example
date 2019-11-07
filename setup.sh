#!/bin/bash

# Download the VOC dataset for INT8 Calibration 
DATA_DIR=/VOCdevkit
if [ -d "$DATA_DIR" ]; then
	echo "$DATA_DIR has already been downloaded"
else
	echo "Downloading VOC dataset"
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
	tar -xf VOCtest_06-Nov-2007.tar
fi