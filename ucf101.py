import os
from numpy import random
import numpy as np
import cv2

RGB_MEAN = np.array([104, 117, 123], np.float32)
FLOW_MEAN = np.array([128.], np.float32)

# def _get_random_segments_start_id(length, num_segments, num_length):
#     avg_length = (length - num_length + 1) // num_segments
#     ids = []
#     for i in range(num_segments):
#         randnum = random.randint(0, avg_length)
#         ids.append(randnum + ids * avg_length)
#     return ids

def _get_equal_length_segments_start_id(length, num_segments, num_length, test_mode):
    avg_length = (length - num_length + 1) // num_segments
    if test_mode:
        firstnum = avg_length // 2
    else:
        firstnum = random.randint(0, avg_length)
    ids = []
    for seg_id in range(num_segments):
        seg_start_id = firstnum + seg_id * avg_length
        ids += range(seg_start_id, seg_start_id+num_length)
    return ids

def _multi_scale_fix_crop(img, is_flow, fix_crop=True, max_distort=1):
    if is_flow:
        scale_ratios = [1.,0.875,0.75]
    else:
        scale_ratios = [1.,0.875,0.75,0.66]
    select_ratio_h_id = random.randint(len(scale_ratios))
    select_ratio_w_id = random.randint(3) - 1 + select_ratio_h_id
    if select_ratio_w_id >= len(scale_ratios):
        select_ratio_w_id = len(scale_ratios) - 1
    elif select_ratio_w_id <= 0:
        select_ratio_w_id = 0
    ratio_h = scale_ratios[select_ratio_h_id]
    ratio_w = scale_ratios[select_ratio_w_id]
    img = cv2.resize(img, None, fx=1./ratio_h, fy=1./ratio_w, interpolation = cv2.INTER_AREA)

    # nine position
    fix_crop_h_id = random.randint(3)
    fix_crop_w_id = random.randint(3)

    h = img.shape[0]
    w = img.shape[1]
    crop_h = crop_w = 224

    h_start_crop_pos = 0
    w_start_crop_pos = 0
    if fix_crop_h_id == 1:
        h_start_crop_pos = (h-crop_h)/2
    elif fix_crop_h_id == 2:
        h_start_crop_pos = h - crop_h
    if fix_crop_w_id == 1:
        w_start_crop_pos = (w-crop_w)/2
    elif fix_crop_w_id == 2:
        w_start_crop_pos = w - crop_w

    img = img[h_start_crop_pos: h_start_crop_pos+crop_h, w_start_crop_pos: w_start_crop_pos+crop_w, :]
    return img

# TODO
def _color_disturbance(img):
    assert False
    return img

# TODO
def _mirror(img, is_flow):
    # mirror
    if is_flow:
        img = img[:,::-1,:]
        # x flow
        img[:,:,0::2] = -img[:,:,0::2]
    else:
        img = img[:,::-1,:]
    return img

def _rgb_transform(img, is_test):
    img = img.astype(np.float32)
    h = img.shape[0]
    w = img.shape[1]
    crop_h = crop_w = 224

    if is_test:
        # crop img
        img = img[(h-crop_h)/2:(h-crop_h)/2+crop_h, (w-crop_w)/2:(w-crop_w)/2+crop_w, :]
        # set mean value
        img = img - RGB_MEAN
        return img
    else:
        # crop img
        img = _multi_scale_fix_crop(img, False)
        # minus average value
        img = img - RGB_MEAN
        # do mirror, p = 0.5
        if bool(random.randint(2)):
            img = _mirror(img, False)
        # img = _color_disturbance(img)
        return img

def _flow_transform(img, is_test):
    img = img.astype(np.float32)
    h = img.shape[0]
    w = img.shape[1]
    crop_h = crop_w = 224
    if is_test:
        img = img[(h-crop_h)/2:(h+crop_h)/2, (w-crop_w)/2:(w+crop_w)/2, :]
        img = img - FLOW_MEAN
        return img
    else:
        # crop img
        img = _multi_scale_fix_crop(img, True)
        # minus average value
        img = img - FLOW_MEAN
        # do mirror, p = 0.5
        if bool(random.randint(2)):
            img = _mirror(img, True)
        # img = _color_disturbance(img)
        return img


def _read_video_segments(video_dir, video_length, num_segments, num_length, frame_type, test_mode):
    assert frame_type == "RGB" or frame_type == "FLOW"
    frame_ids = _get_equal_length_segments_start_id(video_length,
        num_segments, num_length, test_mode)
    if frame_type == "RGB":
        video_block = np.zeros([224, 224, num_segments*num_length*3], dtype=np.float32)
        for i, frame_id in enumerate(frame_ids):
            full_path = os.path.join(video_dir, '%04d.jpg'%frame_id)
            img = cv2.imread(full_path)
            assert isinstance(img, np.ndarray)
            img = _rgb_transform(img, test_mode)
            video_block[:, :, i*3:(i+1)*3] = img
        return video_block
    else:
        video_block = np.zeros([224, 224, num_segments*num_length*2], dtype=np.float32)
        for i, frame_id in enumerate(frame_ids):
            img_x = cv2.imread(os.path.join(video_dir, '%04d_x.jpg'%frame_id), cv2.IMREAD_GRAYSCALE)
            img_y = cv2.imread(os.path.join(video_dir, '%04d_x.jpg'%frame_id), cv2.IMREAD_GRAYSCALE)
            assert isinstance(img_x, np.ndarray)
            assert isinstance(img_y, np.ndarray)
            img_xy = _flow_transform(np.stack([img_x,img_y],axis=2), test_mode)
            video_block[:, :, i*2:(i+1)*2] = img_xy
        return video_block

class reader:
    def __init__(self, root_dir, video_list_fn, batch_size, test=True,
        num_segments=1, num_length=1, frame_type="RGB", full_test=False, full_test_segments=25):
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
        self.batch_size = batch_size
        self.test_mode = test
        self.num_segments = num_segments
        self.num_length = num_length
        self.frame_type = frame_type
        self.full_test = full_test
        if full_test:
            assert self.test_mode == True
        self.full_test_segments = full_test_segments

    def load(self):
        if self.full_test:
            return self._load_full_test()
        data = np.zeros([self.batch_size, 224, 224, (3 if self.frame_type=="RGB" else 2)], dtype=np.float32)
        labels = np.zeros(self.batch_size, dtype=np.int64)
        for i in range(self.batch_size):
            video_item = self.video_list[self.video_id]
            video_dir = video_item[0]
            video_length = video_item[1]
            video_label = video_item[2]
            data[i, :, :, :] = _read_video_segments(
                    os.path.join(self.root_dir, video_dir),
                    video_length, self.num_segments,
                    self.num_length, self.frame_type, self.test_mode)
            labels[i] = video_label
            self.video_id += 1
            if self.video_id >= self.total_video_num:
                self.video_id = 0
        return data, labels

    def get_video_num(self):
        return self.total_video_num

    def _load_full_test(self):
        # no use of num_segments
        if self.video_id >= self.total_video_num:
            return None, None
        assert self.batch_size % self.full_test_segments == 0
        data = np.zeros([self.batch_size, 224, 224, 3], dtype=np.float32)
        labels = np.zeros(self.batch_size, dtype=np.int64)

        for i in range(self.batch_size // self.full_test_segments):
            video_item = self.video_list[self.video_id]
            video_dir = video_item[0]
            video_length = video_item[1]
            video_label = video_item[2]
            # in full test, num segments is set at first
            # [224, 224, full_test_segments*num_length*3/2]
            video_block = _read_video_segments(
                    os.path.join(self.root_dir, video_dir),
                    video_length, self.full_test_segments,
                    self.num_length, self.frame_type, self.test_mode)
            # reshape to [224, 224, full_test_segments, num_length*3/2]
            video_block = video_block.reshape((224, 224, self.full_test_segments, self.num_length*(3 if self.frame_type=="RGB" else 2)))
            # transpose to [full_test_segments, 224, 224, num_length*3/2]
            video_block = np.transpose(video_block, (2, 0, 1, 3))
            data[i*self.full_test_segments:(i+1)*self.full_test_segments, :, :, :] = video_block
            labels[i*self.full_test_segments:(i+1)*self.full_test_segments] = video_label
            self.video_id += 1

            if self.video_id >= self.total_video_num:
                return data, labels
        return data, labels
