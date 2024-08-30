# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam weights는 직접 학습한 모델 또는 yolo가 기본으로 제공해주는 모델의 종류(yolov5가 여기에 해당함), --source는 어떤 종류의 인식을 사용할건지 정함.
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream //카메라 영상을 접속해서 인식 하기 위한 구문

모델을 선정 모델에 맞게끔 속도를 빠르게 하고 싶으면 인식률은 조금 낮아도 yolov5s 속도는 느려도 인식률을 높이고 싶다면 yolov5x
Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse # cmd 동작을 시키기 위한 클래스 관리 대상(epoch, batch_size, ir_initial)
import csv
import os
import platform
import sys
from pathlib import Path
import shutil
from datetime import datetime
import time

import torch 
#facebook에서 제공하는 딥러닝 도구로서, numpy와 효율적인 연동을 지원하는 편리한 도구이다.
#구글에서는 tenorflow에서 개발
#tensorflow나 pytorch나 기본적인 data structure은 tensor이다.
#tensor란 2차원 이상의 array이며, matrix, vector의 일반화된 객체이다.
global_list_num = None
#일반화의 정의 
# vector은 1차원 tensor이다.
# matrix는 2차원 tensor이다.
# 색을 나타내는 RGB는 3차원 tensor이다.
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
    weights=ROOT / "yolov5x.pt",  # model path or triton URL
    source=ROOT / "dataset/train2017/images/val",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_csv=False,  # save results in CSV format
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = save_dir / "predictions.csv"

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            """Writes prediction data for an image to a CSV file, appending if the file exists."""
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            print("p.name = ", p.name)
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
            print("p.stem = ", p.stem)
            s += "%gx%g " % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            list_x1 = []
            list_num = []
            if len(det):
                global global_list_num
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    #print("xyxy = ", xyxy)
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f"{names[c]}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        nn = label
                       # print('nn',nn)
                        ny = nn.split() # splite()공백 제거
                        if ny[0] == 'a1':
                            ny[0] = '가'
                        elif ny[0] == 'a2':
                            ny[0] = '나'
                        elif ny[0] == 'a3':
                            ny[0] = '다'
                        elif ny[0] == 'a4':
                            ny[0] = '라'
                        elif ny[0] == 'a5':
                            ny[0] = '마'
                        

                        elif ny[0] == 'a6':
                            ny[0] = '거'
                        elif ny[0] == 'a7':
                            ny[0] = '너'
                        elif ny[0] == 'a8':
                            ny[0] = '더'
                        elif ny[0] == 'a9':
                            ny[0] = '러'
                        elif ny[0] == 'a10':
                            ny[0] = '머'
                        elif ny[0] == 'a11':
                            ny[0] = '버'
                        elif ny[0] == 'a12':
                            ny[0] = '서'
                        elif ny[0] == 'a13':
                            ny[0] = '어'
                        elif ny[0] == 'a14':
                            ny[0] = '저'
                        

                        elif ny[0] == 'a15':
                            ny[0] = '고'
                        elif ny[0] == 'a16':
                            ny[0] = '노'
                        elif ny[0] == 'a17':
                            ny[0] = '도'
                        elif ny[0] == 'a18':
                            ny[0] = '로'
                        elif ny[0] == 'a19':
                            ny[0] = '모'
                        elif ny[0] == 'a20':
                            ny[0] = '보'
                        elif ny[0] == 'a21':
                            ny[0] = '소'
                        elif ny[0] == 'a22':
                            ny[0] = '오'
                        elif ny[0] == 'a23':
                            ny[0] = '조'
                      

                        elif ny[0] == 'a24':
                            ny[0] = '구'
                        elif ny[0] == 'a25':
                            ny[0] = '누'
                        elif ny[0] == 'a26':
                            ny[0] = '두'
                        elif ny[0] == 'a27':
                            ny[0] = '루'
                        elif ny[0] == 'a28':
                            ny[0] = '무'
                        elif ny[0] == 'a29':
                            ny[0] = '부'
                        elif ny[0] == 'a30':
                            ny[0] = '수'
                        elif ny[0] == 'a31':
                            ny[0] = '우'
                        elif ny[0] == 'a32':
                            ny[0] = '주'
                            
                        elif ny[0] == 'b1':
                            ny[0] = '아'
                        elif ny[0] == 'b2':
                            ny[0] = '바'
                        elif ny[0] == 'b3':
                            ny[0] = '사'
                        elif ny[0] == 'b4':
                            ny[0] = '자'
                        elif ny[0] == 'c1':
                            ny[0] = '배'
                        elif ny[0] == 'd1':
                            ny[0] = '하'
                        elif ny[0] == 'd2':
                            ny[0] = '허'
                        elif ny[0] == 'd3':
                            ny[0] = '호'
                  

                       
                        x1 = int(xyxy[0].item())
                        # y1 = int(xyxy[1].item())
                        # x2 = int(xyxy[2].item())
                        # y1 = int(xyxy[3].item())
                        list_x1.append(x1)
                        list_num.append(ny[0])
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)
                for k in range(len(list_x1)):
                    for j in range(len(list_x1) - 1):
                        if list_x1[j] > list_x1[j + 1]:  # bounding box 좌표를 오름차순으로 정렬
                            list_x1[j], list_x1[j + 1] = list_x1[j + 1], list_x1[j]
                            list_num[j], list_num[j + 1] = list_num[j + 1], list_num[j]

                list_numm = "".join(list_num)
                list_num_len = len(list_numm)
                

                total_score = 0
                if (len(list_numm) > 6 and 
                    (list_numm[0] in '0123456789') and
                    (list_numm[1] in '0123456789') and
                    (list_numm[2] in '가나다라마바사아자하거너더러머버서어저허고노도로모보소오조호구누두루무부수우주') and
                    (list_numm[3] in '0123456789') and
                    (list_numm[4] in '0123456789') and
                    (list_numm[5] in '0123456789') and
                    (list_numm[6] in '0123456789') and
                    (list_num_len in [7, 8, 9])):
                    global_list_num = list_numm
                elif (len(list_numm) > 7 and 
                    (list_numm[0] in '0123456789') and
                    (list_numm[1] in '0123456789') and
                    (list_numm[2] in '0123456789') and
                    (list_numm[3] in '가나다라마바사아자하거너더러머버서어저허고노도로모보소오조호구누두루무부수우주') and
                    (list_numm[4] in '0123456789') and
                    (list_numm[5] in '0123456789') and
                    (list_numm[6] in '0123456789') and
                    (list_numm[7] in '0123456789') and
                    (list_num_len in [7, 8, 9])):
                    global_list_num = list_numm    

                print("p.stem=", p.stem)  # 사진번호
                print("global_list=", global_list_num)  # 차량번호
                #print("차선=", p.stem.split("=")[1])
                try:
                    # Original file renaming code
                    now = datetime.now()
                    datetime_1 = now.strftime("%Y-%m-%d-%H-%M-%S")
                    datetime_2 = datetime_1.split(sep = "-")
            
                    year = int(datetime_2[0])
                    month = int(datetime_2[1])
                    day = int(datetime_2[2])
                    hour = int(datetime_2[3])
                    
                    original_path = Path(p)
                    lane = p.stem.split("=")[1]
                    new_name = f"{year}{month}{day}{hour}-{global_list_num}-{lane}.jpg" # New image name
                    new_path = str(original_path.parent / new_name)  # New image path
                    shutil.move(str(original_path), str(new_path))  # Rename file
                    #print(f"Renamed {save_path} to {new_path}")
                except:
                    original_path = Path(p)
                    lane = 000
                    new_name = f"{year}{month}{day}{hour}-{global_list_num}-{lane}.jpg" # New image name
                    new_path = str(original_path.parent / new_name)  # New image path
                    shutil.move(str(original_path), str(new_path))  # Rename file
    
    
            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)
            

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    """Parses command-line arguments for YOLOv5 detection, setting inference options and model configurations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "C:/Users/Administrator/Desktop/exe/20240617project/yolov5-master/runs/number_best/best.pt", help="model path or triton URL") # yolo 모델 선정
    parser.add_argument("--source", type=str, default=ROOT / "C:/Users/Administrator/Desktop/exe/20240617project/yolov5-master/runs/detect/exp/crops", help="file/dir/URL/glob/screen/0(webcam)") # 학습 데이터 read 경로
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path") # class 경로 및 파일
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w") # 학습사이즈 얼마로 줄거냐
    parser.add_argument("--conf-thres", type=float, default=0.125, help="confidence threshold") # 인식률 설정  기존0.25
    parser.add_argument("--iou-thres", type=float, default=0.125, help="NMS IoU threshold") # 인식률 설정  기존0.45
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image") # 이미지 몇장까지
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu") # cpu를 사용 할 지 cuda를 사용 할 지 설정
    parser.add_argument("--view-img", action="store_true", help="show results") # 이미지 view
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes") # crop box 설정
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos") # video관련
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name") # 인식 데이터 결과가 저장 되는 곳
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=1, type=int, help="bounding box thickness (pixels)") # box 굵기
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=True, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    """Executes YOLOv5 model inference with given options, checking requirements before running the model."""
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
