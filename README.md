# MAFAT-Satellite-Vision-Challenge

## Bad images in annotated data
For some reasons the patchfiying of the images introduced some faulty images:
 - 6298_10240_2560__640__320___0
 - 15031_2560_1280__640__0___320
 - 5431_3840_0__640__320___640
 - 1140_5120_3840__640__0___320
 - 1107_1280_0__640__0___320
 - 14712_3840_11520__640__0___640
 - 19344_0_10240__640__0___640
 - 15098_1280_0__640__320___640
 - 9354_10240_3840__640__320___320
 - 18169_0_0__640__640___320
 - 557_6400_5120__640__320___0
 
They are removed in the patchfied dataset. 
However if there is a problem in the training pipeline consider deleting these images with their annotation files.