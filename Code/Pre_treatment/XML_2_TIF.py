from xml import sax
import os
import cv2
import numpy as np
class Box_Handler(sax.ContentHandler):  # 定义自己的handler类，继承sax.ContentHandler
    def __init__(self):
        sax.ContentHandler.__init__(self)  # 弗父类和子类都需要初始化（做一些变量的赋值操作等）
        self.CurrentData = ""
        self.tag = ""
        self.xmin = ""
        self.ymin = ""
        self.xmax = ""
        self.ymax = ""
        self.temp = []
        self.box = []



    def startElement(self, name, attrs):  # 遇到<tag>标签时候会执行的方法，这里的name，attrs不用自己传值的（这里其实是重写）
        self.CurrentData = name

    def endElement(self, name):
    # 遇到</tag>执行的方法，name不用自己传值（重写）
    # print "endElement"
        if name == "xmin":
            self.temp.append(int(self.xmin))
        elif name == "ymin":
            self.temp.append(int(self.ymin))
        elif name == "xmax":
            self.temp.append(int(self.xmax))
        elif name == "ymax":
            self.temp.append(int(self.ymax))
        elif name == "object":
            self.box.append(self.temp)
            self.temp = []
        elif name == "annotation":
            global Box
            Box = self.box
        self.CurrentData = ""
        return self.box
    def characters(self, content):  # 获取标签内容
        if self.CurrentData == "name":
            self.tag = content
        elif self.CurrentData == "xmin":
            self.xmin = content
        elif self.CurrentData == "ymin":
            self.ymin = content
        elif self.CurrentData == "xmax":
            self.xmax = content
        elif self.CurrentData == "ymax":
            self.ymax = content

def read_box(path):
    '''读取xml文件，并返回box的列表[[xmin,ymin,xmax,ymax], [xmin,ymin,xmax,ymax]]'''
    parser = sax.make_parser()  # 创建一个 XMLReader
    parser.setFeature(sax.handler.feature_namespaces, 0)  # turn off namepsaces
    Handler = Box_Handler()  # 重写 ContextHandler
    parser.setContentHandler(Handler)
    parser.parse(path)
    return Box

def get_undo_list(path, flag):
    '''获取目标路径中的所有文件，并按顺序排好，返回未出来文件路径的列表'''
    all_xml = os.listdir(path)
    #all_xml = sorted(all_xml, key=lambda x: int(x.split('.')[-2]))
    print(all_xml)
    undo_xml = []
    for i in range(len(all_xml)):
        name = all_xml[i].split('.')[-2]

        if flag in name:
            pass
        else:
            undo_xml.append(path + "\\" + all_xml[i])
    undo_xml = sorted(undo_xml, key=lambda x: int(x.split('.')[-2].split('\\')[-1]))
    return undo_xml

def grab_img(soure_path, taget_file, box, flag):
    '''读取图片中的所有box并保存'''
    img = cv2.imread(soure_path)
    for b in box:
        temp = img[b[1]:b[3], b[0]:b[2]]
        temp = cv2.resize(temp, (128, 128), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(taget_file + "\\" + str(flag) + '.tif', temp)
        flag += 1
    return flag

if __name__ == "__main__":
    # 1-循环读取所有文件夹-按顺序
    # 2-循环读取所有xml文件-按顺序
    # 3-读取所有Box
    # 4-从对应图像中读取图像，并另存为
    # 5-已经读取后的xml名字修改

    Source_root = "D:\Project_Data\First\clear_data"
    Target_root = "D:\Project_Data\First\Box_data"

    for k in range(9, 10):
        pa = 'class_%d' % (k)
        Source_file = Source_root + '\class_' + str(k) + '_xml'
        Img_file = Source_root + '\class_' + str(k)
        Target_file = Target_root + '\class_' + str(k)
        if not os.path.exists(Target_file):
            os.makedirs(Target_file, exist_ok=True)
        # if pa == 'test_1':
        #     continue
        flag = len(os.listdir(Target_file))   # 这个flag用来记录目标文件夹中的图片数量，从而按顺序命名
        undo_xml = get_undo_list(Source_file, '_ok')

        for i in range(len(undo_xml)):
             Box = read_box(undo_xml[i])
             # print(Box)
             Img_path = Img_file + "\\" + undo_xml[i].split(".")[-2].split("\\")[-1] + ".tif"
             print(Img_path)
             flag = grab_img(Img_path, Target_file, Box, flag)
             os.rename(undo_xml[i], undo_xml[i].split('.')[-2] + '_ok.xml')






