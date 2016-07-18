import os
from numpy import random
import numpy as np
import cv2
import threading
import Queue

RGB_MEAN = np.array([104, 117, 123], np.float32)
FLOW_MEAN = 128.

def _get_equal_length_segments_start_id(length, num_segments, num_length, is_test):
    avg_length = (length - num_length + 1) // num_segments
    if is_test:
        firstnum = avg_length // 2
    else:
        firstnum = random.randint(0, avg_length)
    ids = []
    for seg_id in range(num_segments):
        seg_start_id = firstnum + seg_id * avg_length
        ids += range(seg_start_id, seg_start_id+num_length)
    return ids

def _multi_scale_fix_crop(img, is_flow):
    img = img.astype(np.float32)
    if is_flow:
        scale_ratios = [1.,0.875,0.75]
    else:
        scale_ratios = [1.,0.875,0.75,0.66]
    ratio_h = ratio_w = scale_ratios[random.randint(len(scale_ratios))]
    img = cv2.resize(img, None, fx=1./ratio_h, fy=1./ratio_w, interpolation = cv2.INTER_LINEAR)

    h = img.shape[0]
    w = img.shape[1]
    crop_h = crop_w = 224
    fix_crop_id = random.randint(5)
    if fix_crop_id == 1:
        h_start_crop_pos = 0
        w_start_crop_pos = 0
    elif fix_crop_id == 2:
        h_start_crop_pos = 0
        w_start_crop_pos = w - crop_w
    elif fix_crop_id == 3:
        h_start_crop_pos = h - crop_h
        w_start_crop_pos = 0
    elif fix_crop_id == 4:
        h_start_crop_pos = h - crop_h
        w_start_crop_pos = w - crop_w
    else:
        h_start_crop_pos = (h - crop_h) // 2
        w_start_crop_pos = (w - crop_w) // 2
    img = img[h_start_crop_pos: h_start_crop_pos+crop_h, w_start_crop_pos: w_start_crop_pos+crop_w, :]
    return img

def _mirror(img, is_flow):
    # mirror
    if is_flow:
        img = img[:,::-1,:]
        # x flow
        img[:,:,0::2] = -img[:,:,0::2]
    else:
        img = img[:,::-1,:]
    return img

def _minus_mean(img, is_flow):
    if is_flow:
        return img - FLOW_MEAN
    else:
        for i in range(0, img.shape[2], 3):
            img[:, :, i:i+3] -= RGB_MEAN
        return img

def _transform(img, is_test, is_flow):
    img = img.astype(np.float32)
    h = img.shape[0]
    w = img.shape[1]
    crop_h = crop_w = 224

    if is_test:
        img = img[(h-crop_h)/2:(h-crop_h)/2+crop_h, (w-crop_w)/2:(w-crop_w)/2+crop_w, :]
        img = _minus_mean(img, is_flow)
        return img
    else:
        img = _multi_scale_fix_crop(img, is_flow)
        img = _minus_mean(img, is_flow)
        if bool(random.randint(2)):
            img = _mirror(img, is_flow)
        return img

def _read_video_segments(video_dir, video_length, num_segments, num_length, frame_type, is_test):
    assert frame_type == "RGB" or frame_type == "FLOW"
    is_flow = True if frame_type == "FLOW" else False
    num_channel = 2 if frame_type == "FLOW" else 3
    frame_ids = _get_equal_length_segments_start_id(video_length, num_segments, num_length, is_test)

    video_block = np.zeros([224, 224, num_segments*num_length*num_channel], dtype=np.float32)
    # read all images
    images = []
    if frame_type == "RGB":
        for frame_id in frame_ids:
            img = cv2.imread(os.path.join(video_dir, '%04d.jpg'%frame_id))
            assert isinstance(img, np.ndarray)
            images.append(img)
    else:
        for frame_id in frame_ids:
            img_x = cv2.imread(os.path.join(video_dir, '%04d_x.jpg'%frame_id), cv2.IMREAD_GRAYSCALE)
            assert isinstance(img_x, np.ndarray)
            img_y = cv2.imread(os.path.join(video_dir, '%04d_y.jpg'%frame_id), cv2.IMREAD_GRAYSCALE)
            assert isinstance(img_y, np.ndarray)
            images.append(np.stack((img_x, img_y), axis=2))
    images = np.concatenate(images,axis=2)
    images = _transform(images, is_test, is_flow)
    return images

class reader:
    def __init__(self, root_dir, video_list_fn, frame_type,
            batch_size, num_length, num_segments,
            is_test,
            mode="NORMAL",
            queue_num=3):
        # video list init
        video_list = []
        with open(video_list_fn) as f:
            lines = f.readlines()
            for line in lines:
                splits = line.split(' ')
                splits = [split.strip() for split in splits]
                video_list.append((splits[0], int(splits[1]), int(splits[2])))
        self.video_list = video_list
        self.total_video_num = len(self.video_list)
        assert self.total_video_num > 0
        self.video_id = 0

        self.root_dir = root_dir
        assert frame_type == "RGB" or frame_type=="FLOW"
        self.frame_type = frame_type
        self.batch_size = batch_size
        self.num_segments = num_segments
        self.num_length = num_length
        self.is_test = is_test
        self.mode=mode

        assert mode=="NORMAL" or mode=="FULLTEST" or mode=="SEQ"
        if mode=="NORMAL":
            self.full_test = False
            self.is_seq = False
        elif mode=="FULLTEST":
            assert self.is_test == True
            self.full_test = True
            self.is_seq = False
        else:
            self.is_seq = True

        self.mutex = threading.Lock()
        self.threads = [threading.Thread(target=self._queue_load) for i in range(queue_num)]
        self.queue = Queue.Queue(maxsize = queue_num)
        for t in self.threads:
            t.setDaemon(True)
            t.start()

    def get(self):
        return self.queue.get()

    def get_video_num(self):
        return self.total_video_num

    def _queue_load(self):
        while(True):
            data_labels = self._load()
            self.queue.put(data_labels)

    def _load(self):
        assert self.mode=="NORMAL" or self.mode=="FULLTEST" or self.mode=="SEQ"

        self.mutex.acquire()
        ids = []
        for i in range(self.batch_size):
            ids.append(self.video_id)
            self.video_id += 1
            if self.video_id >= self.total_video_num:
                self.video_id = 0
        self.mutex.release()

        img_channel = (3 if self.frame_type=="RGB" else 2)
        if self.mode=="NORMAL":
            data = np.zeros([self.batch_size, 224, 224, self.num_segments*self.num_length*img_channel], dtype=np.float32)
            labels = np.zeros(self.batch_size, dtype=np.int64)
        else:
            data = np.zeros([self.batch_size*self.num_segments, 224, 224, self.num_length*img_channel], dtype=np.float32)
            labels = np.zeros(self.batch_size*self.num_segments, dtype=np.int64)

        for i, vid in enumerate(ids):
            video_item = self.video_list[vid]
            video_dir = video_item[0]
            video_length = video_item[1]
            video_label = video_item[2]
            video_block = _read_video_segments(
                    os.path.join(self.root_dir, video_dir),
                    video_length, self.num_segments,
                    self.num_length, self.frame_type, self.is_test)
            if self.mode == "NORMAL":
                data[i, :, :, :] = video_block
                labels[i] = video_label
            else:
                video_block = video_block.reshape((224, 224, self.num_segments, self.num_length*img_channel))
                # to [num_segments, 224, 224, num_length*3/2]
                video_block = np.transpose(video_block, (2, 0, 1, 3))
                data[i*self.num_segments:(i+1)*self.num_segments, :, :, :] = video_block
                labels[i*self.num_segments:(i+1)*self.num_segments] = video_label
        return data, labels
