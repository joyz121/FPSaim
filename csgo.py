import os
import sys
import time
import cv2
import mss
import numpy as np
from pathlib import Path
import torch
from pynput import keyboard
from pynput import mouse
from ctypes import CDLL
from pypiddemo import pid
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
from multiprocessing import set_start_method, Queue, Process
rect_h=320
rect_w=320
win_h=1080
win_w=1920
relatively_posi={}

def init(
    weights=ROOT / 'yolov5n.pt',  # model path or triton URL
    data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
    imgsz=(320, 320),  # inference size (height, width)
    device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
):
    # Load model
    device = select_device(device=device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride,pt = model.stride, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    bs = 1  # batch_size
        # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    return model

# 矩形框坐标转换到显示器坐标
def rect2realpos(position):
    rect_x=position[0]
    rect_y=position[1]
    realpos_y=(win_h-rect_h)/2+rect_y
    realpos_x=(win_w-rect_w)/2+rect_x
    return (realpos_x,realpos_y)
# 推断
def Inference(source,model):
    names=model.names
    im0s=source
    im = im0s
    bestmatch=20000
    im=im.transpose((2, 0, 1))[::-1]
    im = np.ascontiguousarray(im)
    dt =(Profile(), Profile(), Profile())
    with dt[0]:
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
    # Inference
    with dt[1]:
        pred = model(im)
    # NMS
    with dt[2]:
        pred = non_max_suppression(pred,conf_thres=0.3,iou_thres=0.45,classes=0)
    for i, det in enumerate(pred):  # per image
        im0=im0s.copy()
        annotator = Annotator(im0, line_width=3, example=str(names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                center_x=(int(xyxy[0].item())+int(xyxy[2].item()))/2
                center_y=int(xyxy[1].item())+(int(xyxy[3].item())-int(xyxy[1].item()))/5
                real_posi=rect2realpos(position=(center_x,center_y))
                distance=np.sqrt(np.power((win_w/2-real_posi[0]),2)+np.power((win_h/2-real_posi[1]),2))
                if distance<bestmatch:
                    bestmatch=distance
                    relatively_posi[0]=real_posi[0]-win_w/2
                    relatively_posi[1]=real_posi[1]-win_h/2
                    #target_posi=real_posi
                    target_posi=relatively_posi
                #label = names[c]
                annotator.box_label(xyxy, color=colors(c, True))
        else:
            target_posi=(0,0)#no target
    im0 = annotator.result()
    return im0,target_posi


flag=False
flag_key=False
def on_press(key):
    global flag_key
    if key == keyboard.Key.ctrl_l:
        if flag_key==True:
            flag_key=False
        else:
            flag_key=True

def on_press1(x,y,button,pressed):
    global flag
    if pressed:
        flag=True
    else:
        flag=False


def mouse_xy(gm,x, y, abs_move = False):
    return gm.Mach_Move(int(x), int(y), abs_move)
def pid_move(gm,mousepidx,mousepidy,x,y):
    x_t=mousepidx.pid_inc(0,x)
    y_t=mousepidy.pid_inc(0,y)
    mouse_xy(gm,int(x+x_t), int(y+y_t))

def detect(queue):
    sct = mss.mss()
    cout=0
    fps=0
    model=init()
    monitor={"top":int((win_h-rect_h)/2),"left":int((win_w-rect_w)/2.0),"width":rect_w,"height":rect_h}
    start=time.time()
    while True:
        frame=np.array(sct.grab(monitor))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        #check_requirements(exclude=('tensorboard', 'thop'))
        frame,target_posi=Inference(source=frame,model=model)
        #print(target_posi)
        if target_posi!=(0,0):#have target
            queue.put(target_posi)
        # show screenshot
        cv2.namedWindow('win',cv2.WINDOW_FULLSCREEN)
        cv2.resizeWindow('win',frame.shape[1], frame.shape[0])
        # write fps
        cv2.putText(frame,"%.2f fps"%fps,(5,30),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,255),2)
        cv2.imshow("win",frame)
        cout+=1
        if cout==10:
            end=time.time()
            seconds=end-start
            fps=float(cout)/seconds
            start=time.time()
            cout=0
        if cv2.waitKey(1) & 0xff == ord('0'):
            break
def mouse_ctrl(queue):
    gm = CDLL('./ghub_mouse.dll')
    gmok = gm.Agulll()
    if not gmok:
        print('未安装ghub或者lgs驱动!!!')
    else:
        print('鼠标初始化成功!')
    global flag_key
    mousepidx=pid(0.1,0,0)
    mousepidy=pid(0.1,0,0)
    key_listener=keyboard.Listener(on_press=on_press)
    mouse_listener=mouse.Listener(on_click=on_press1)
    key_listener.start()#键盘监听
    mouse_listener.start()#鼠标监听
    while True:
        if queue.empty()is True:
            continue
        pos=queue.get()
        if flag_key:
            pid_move(gm,mousepidx,mousepidy,pos[0],pos[1])
if __name__=="__main__":
    set_start_method('spawn')
    q = Queue()
    p_d = Process(target=detect, args=(q,))
    p_m = Process(target=mouse_ctrl, args=(q,))
    p_d.start()
    p_m.start()