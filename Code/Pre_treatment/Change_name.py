import os
Source_root = "D:\Program_Data\First\clear_data"
for i in range(8):
    pa = "\\" + "class_" + str(i) + "_xml"
    Source_file = Source_root + pa
    dir = os.listdir(Source_file)
    for d in dir:
        num = d.split("\\")[-1].split("_")[0]
        os.rename(Source_file + "\\" + d, Source_file + "\\" + num + ".xml")