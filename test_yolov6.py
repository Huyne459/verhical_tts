from my_yolov6 import my_yolov6
import cv2
import csv

yolov6_model = my_yolov6("weights/yolov6s.pt","cpu","data/coco.yaml", 640, True)


# define a video capture object
vid = cv2.VideoCapture('/home/huy/Desktop/TL-tech/Verhical_count/data/Singapore/20220929_103214_062.MP4')
width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('your_video.avi', fourcc, 10.0, size)
count = 0
while (True):
    middle_line_position = 850
    up_line_position = middle_line_position - 30
    down_line_position = middle_line_position + 30
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    try:
        ih, iw, channels = frame.shape
        frame, len_det, det, up_list, down_list = yolov6_model.infer(frame, conf_thres=0.4, iou_thres=0.45)
        # print('up_list:', up_list)
        # print('down_list:', down_list)
    except:
        break
    print('count: ', count)
    count += 1
    # print('boxs: ', boxs)

    # Display the resulting frame
    cv2.line(frame, (0, middle_line_position), (iw, middle_line_position), (255, 0, 255), 2)
    cv2.line(frame, (0, up_line_position), (iw, up_line_position), (0, 0, 255), 2)
    cv2.line(frame, (0, down_line_position), (iw, down_line_position), (0, 0, 255), 2)
    out.write(frame)
    cv2.imshow('frame', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

with open("data.csv", 'w') as f1:
    cwriter = csv.writer(f1)
    names = [ 'Direction','person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]
    cwriter.writerow(names)
    up_list.insert(0, "Up")
    down_list.insert(0, "Down")
    cwriter.writerow(up_list)
    cwriter.writerow(down_list)
    f1.close()
# After the loop release the cap object
vid.release()
out.release()
# Destroy all the windows
cv2.destroyAllWindows()