import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time


gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# gpu = torch.device("cpu")
# print(gpu)


def plot_detections(img, detections, with_keypoints=True):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.grid(False)
    ax.imshow(img)

    if isinstance(detections, torch.Tensor):
        detections = detections.cpu().numpy()

    if detections.ndim == 1:
        detections = np.expand_dims(detections, axis=0)

    print("Found %d faces" % detections.shape[0])

    for i in range(detections.shape[0]):
        ymin = detections[i, 0] * img.shape[0]
        xmin = detections[i, 1] * img.shape[1]
        ymax = detections[i, 2] * img.shape[0]
        xmax = detections[i, 3] * img.shape[1]

        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=1, edgecolor="r", facecolor="none",
                                 alpha=detections[i, 16])
        ax.add_patch(rect)

        if with_keypoints:
            for k in range(6):
                kp_x = detections[i, 4 + k*2    ] * img.shape[1]
                kp_y = detections[i, 4 + k*2 + 1] * img.shape[0]
                circle = patches.Circle((kp_x, kp_y), radius=0.5, linewidth=1,
                                        edgecolor="lightskyblue", facecolor="none",
                                        alpha=detections[i, 16])
                ax.add_patch(circle)

    plt.show()

def plot_cv2_detections(img, detections):
    if isinstance(detections, torch.Tensor):
        detections = detections.cpu().numpy()

    if detections.ndim == 1:
        detections = np.expand_dims(detections, axis=0)

    for i in range(detections.shape[0]):
        ymin = detections[i, 0] * img.shape[0]
        xmin = detections[i, 1] * img.shape[1]
        ymax = detections[i, 2] * img.shape[0]
        xmax = detections[i, 3] * img.shape[1]

    cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 255))
    cv2.imshow("img", img)
    cv2.waitKey(1)

def preprocess_face_detect(image, frameSize, output_size):
    if frameSize[0] > frameSize[1]:
        mid_pt = frameSize[0] // 2
        square_len = frameSize[1]
        short_side = mid_pt - square_len // 2
        long_side = mid_pt + square_len // 2
        img_crop = image[:, int(short_side):int(long_side)]
    elif frameSize[0] < frameSize[1]:
        mid_pt = frameSize[1] // 2
        square_len = frameSize[0]
        short_side = mid_pt - square_len // 2
        long_side = mid_pt + square_len // 2
        img_crop = image[int(short_side):int(long_side), :]

    image_resize = cv2.resize(img_crop,
                              (output_size[0], output_size[1]))
    scale_ratio = image.shape[0] / output_size[0]

    return image_resize, img_crop, scale_ratio


from blazeface import BlazeFace

net = BlazeFace().to(gpu)
net.load_weights("blazeface.pth")
net.load_anchors("anchors.npy")

net.min_score_thresh = 0.75
net.min_suppression_threshold = 0.3

def predict_on_one_img():
    img = cv2.imread("1face.png")
    img = cv2.resize(img, (128,128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    start = time.time()
    detections = net.predict_on_image(img)
    end = time.time()
    print('one image time: ', end - start)
    last = end - start
    plot_detections(img, detections)
    return last

def predict_on_batch():
    filenames = ["1face.png", "3faces.png", "4faces.png"]

    x = np.zeros((len(filenames), 128, 128, 3), dtype=np.uint8)

    for i, filename in enumerate(filenames):
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x[i] = img
    start = time.time()
    detections = net.predict_on_batch(x)
    end = time.time()
    print('batch time: ', end - start)
    plot_detections(x[0], detections[0])
    last = end - start
    return last


def predict_on_camera(cap_idx=1):
    cap = cv2.VideoCapture(cap_idx)
    frameSize = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    hasFrame, frame = cap.read()

    # while True:
    while hasFrame:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img, _, _ = preprocess_face_detect(img, frameSize, (128, 128))
        detections = net.predict_on_image(img)
        plot_cv2_detections(img, detections)
        hasFrame, frame = cap.read()

if __name__ == '__main__':

    # batch = predict_on_batch()
    # one_img = predict_on_one_img()
    # two = (batch - one_img) / 2
    # print(two)
    predict_on_camera(1)
