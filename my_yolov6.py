import torch
import numpy as np
import math
import cv2

from tracker import *

from yolov6.utils.events import load_yaml
from yolov6.layers.common import DetectBackend
from yolov6.data.data_augment import letterbox
from yolov6.utils.nms import non_max_suppression

tracker = EuclideanDistTracker()

def find_center(x1, y1, x2, y2):

    cx = (x2+x1)/2
    cy = (y2+y1)/2
    return int(cx), int(cy)

middle_line_position = 850
up_line_position = middle_line_position - 30
down_line_position = middle_line_position + 30

temp_up_list = []
temp_down_list = []
up_list = [0] * 80
down_list = [0] * 80

class my_yolov6():
    def __init__(self, weights, device, yaml, img_size, half):
        self.__dict__.update(locals())

        # Init model
        self.device = device
        self.img_size = img_size
        cuda = self.device != 'cpu' and torch.cuda.is_available()
        self.device = torch.device(f'cuda:{device}' if cuda else 'cpu')
        self.model = DetectBackend(weights, device=self.device)
        self.stride = self.model.stride
        self.class_names = load_yaml(yaml)['names']
        self.img_size = self.check_img_size(self.img_size, s=self.stride)  # check image size

        # Half precision
        if half & (self.device.type != 'cpu'):
            self.model.model.half()
            self.half = True
        else:
            self.model.model.float()
            self.half = False

        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, *self.img_size).to(self.device).type_as(
                next(self.model.model.parameters())))  # warmup

        # Switch model to deploy status
        self.model_switch(self.model.model, self.img_size)

    @staticmethod
    def plot_box_and_label(image, lw, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        # Add one xyxy box to image with label
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))

        if label:
            tf = max(lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=lw / 6, thickness=tf)[0]  # text width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 6, txt_color,
                        thickness=tf, lineType=cv2.LINE_AA)

    @staticmethod
    def rescale(ori_shape, boxes, target_shape):
        '''Rescale the output to the original image shape'''
        ratio = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
        padding = (ori_shape[1] - target_shape[1] * ratio) / 2, (ori_shape[0] - target_shape[0] * ratio) / 2

        boxes[:, [0, 2]] -= padding[0]
        boxes[:, [1, 3]] -= padding[1]
        boxes[:, :4] /= ratio

        boxes[:, 0].clamp_(0, target_shape[1])  # x1
        boxes[:, 1].clamp_(0, target_shape[0])  # y1
        boxes[:, 2].clamp_(0, target_shape[1])  # x2
        boxes[:, 3].clamp_(0, target_shape[0])  # y2

        return boxes

    @staticmethod
    def make_divisible(x, divisor):
        # Upward revision the value x to make it evenly divisible by the divisor.
        return math.ceil(x / divisor) * divisor

    def check_img_size(self, img_size, s=32, floor=0):
        """Make sure image size is a multiple of stride s in each dimension, and return a new shape list of image."""
        if isinstance(img_size, int):  # integer i.e. img_size=640
            new_size = max(self.make_divisible(img_size, int(s)), floor)
        elif isinstance(img_size, list):  # list i.e. img_size=[640, 480]
            new_size = [max(self.make_divisible(x, int(s)), floor) for x in img_size]
        else:
            raise Exception(f"Unsupported type of img_size: {type(img_size)}")

        if new_size != img_size:
            print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
        return new_size if isinstance(img_size,list) else [new_size]*2

    def model_switch(self, model, img_size):
        ''' Model switch to deploy status '''
        from yolov6.layers.common import RepVGGBlock
        for layer in model.modules():
            if isinstance(layer, RepVGGBlock):
                layer.switch_to_deploy()

    def precess_image(self,img_src, img_size, stride, half):
        '''Process image before image inference.'''
        image = letterbox(img_src, img_size, stride=stride)[0]
        # Convert
        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image = torch.from_numpy(np.ascontiguousarray(image))
        image = image.half() if half else image.float()  # uint8 to fp16/32
        image /= 255  # 0 - 255 to 0.0 - 1.0

        return image, img_src

    def count_vehicle(self, box_id, img):
        global temp_up_list, temp_down_list, up_list, down_list

        # print('up_list1:', up_list)
        # print('down_list1:', down_list)

        x, y, w, h, id, index = box_id
        # print('box_id: ', box_id)

        # Find the center of the rectangle for detection
        center = find_center(x, y, w, h)
        # print(center)
        ix, iy = center

        # Find the current position of the vehicle
        if (iy > up_line_position) and (iy < middle_line_position):
            print("hello")
            if id not in temp_up_list:
                temp_up_list.append(id)

        elif iy < down_line_position and iy > middle_line_position:
            print("hello1")
            if id not in temp_down_list:
                temp_down_list.append(id)

        elif iy < up_line_position:
            if id in temp_down_list:
                print("hello2")
                temp_down_list.remove(id)
                up_list[index] = up_list[index] + 1

        elif iy > down_line_position:
            if id in temp_up_list:
                print("hello3")
                temp_up_list.remove(id)
                down_list[index] = down_list[index] + 1

        # Draw circle in the middle of the rectangle
        cv2.circle(img, center, 2, (0, 0, 255), -1)  # end here
        # cv2.putText(img, str(id), center, 0, 1, (0, 0, 256))
        # print('up_list2:', up_list)
        # print('down_list2:', down_list)

    # color = (128, 128, 128), txt_color = (255, 255, 255)
    def infer(self, source, conf_thres=0.4, iou_thres=0.45, classes=None, agnostic_nms=False, max_det=1000):
        img, img_src = self.precess_image(source, self.img_size, self.stride, self.half)
        img = img.to(self.device)

        if len(img.shape) == 3:
            img = img[None]
            # expand for batch dim

        pred_results = self.model(img)
        det = non_max_suppression(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]
        detection = []
        if len(det):
            det[:, :4] = self.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
            for *xyxy, conf, cls in reversed(det):
                class_num = int(cls)  # integer class
                label = f'{self.class_names[class_num]} {conf:.2f}'
                self.plot_box_and_label(img_src, max(round(sum(img_src.shape) / 2 * 0.003), 2), xyxy, label, color=(255,0,0))
                x, y, w, h = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                # print("xyxy: ", xyxy)
                # print(type(xyxy))
                if h > 800:
                    detection.append([x, y, w, h, class_num])

            img_src = np.asarray(img_src)
            try:
                boxes_ids = tracker.update(detection)
                for box_id in boxes_ids:
                    self.count_vehicle(box_id, img_src)
            except:
                boxes_ids = []


        return img_src, len(det), reversed(det), up_list, down_list



