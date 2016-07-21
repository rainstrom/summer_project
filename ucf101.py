import os
from numpy import random
import numpy as np
import cv2
import threading
import Queue
from time import time

RGB_MEAN = np.array([104, 117, 123], np.float32)
FLOW_MEAN = np.array([128, 128], np.float32)

class Transformer:
    def __init__(self, is_test, is_flow, num_segments, num_length):
        self.is_test = is_test
        self.is_flow = is_flow
        self.crop_h = self.crop_w = 224
        self.mean_value = np.tile(FLOW_MEAN if is_flow else RGB_MEAN, num_segments*num_length)
        if is_flow:
            self.scale_ratios = [1.,0.875,0.75]
        else:
            self.scale_ratios = [1.,0.875,0.75,0.66]

    def transform(self, images):
        images = images.astype(np.float32)

        if self.is_test:
            h = images.shape[0]
	    w = images.shape[1]
            h_off = (h - self.crop_h) // 2
            w_off = (w - self.crop_w) // 2
            images = images[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w, :]
            images -= self.mean_value
        else:
            # multi scale fix crop
            images = self._multi_scale_fix_crop(images)
            # minus mean value
            images -= self.mean_value
            # mirror
            if bool(random.randint(2)):
                images = images[:,::-1,:]
                if self.is_flow:
                    images[:,:,0::2] = -images[:,:,0::2]
        return images

    def _multi_scale_fix_crop(self, images):
        scale_ratio = self.scale_ratios[random.randint(len(self.scale_ratios))]
        croph = int(self.crop_h * scale_ratio)
        cropw = int(self.crop_w * scale_ratio)
        h = images.shape[0]
        w = images.shape[1]
        fix_crop_id = random.randint(5)

        if fix_crop_id == 1:
            cropped_images = images[:croph, :cropw, :]
        elif fix_crop_id == 2:
            cropped_images = images[:croph, -cropw:, :]
        elif fix_crop_id == 3:
            cropped_images = images[-croph:, :cropw, :]
        elif fix_crop_id == 4:
            cropped_images = images[-croph:, -cropw:, :]
        else:
            cropped_images = images[(h-croph)//2:(h-croph)//2+croph, (w-cropw)//2:(w-cropw)//2+cropw, :]
        if scale_ratio != 1.:
            cropped_images = cv2.resize(cropped_images, dsize=(self.crop_h, self.crop_w), interpolation = cv2.INTER_LINEAR)
        return cropped_images

class reader:
    def __init__(self,
            root_dir, video_list_fn, frame_type,
            batch_size, num_length, num_segments,
            is_test,
            mode="NORMAL",
            queue_num=3):
        # processing video_list
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

        # set root dir and frame type and something related to frame_type
        self.root_dir = root_dir
        assert frame_type == "RGB" or frame_type=="FLOW"
        self.frame_type = frame_type
        self.is_flow = (True if frame_type=="FLOW" else False)
        self.img_channel = (2 if self.is_flow else 3)

        # batch_size, num_length, num_segments, is_test
        self.batch_size = batch_size
        self.num_segments = num_segments
        self.num_length = num_length
        self.is_test = is_test
        self.mode=mode
        self.transformer = Transformer(self.is_test, self.is_flow, num_segments, num_length)

        # processing mode related info
        if mode=="NORMAL":
            self._load = self._normal_load
        elif mode=="FULLTEST":
            assert self.is_test == True
            self._load = self._fulltest_load
        elif mode=="SEQ":
            self._load = self._seq_load
        else:
            assert False

        # prefetch queue
        # self.mutex = threading.Lock()
        # self.threads = [threading.Thread(target=self._queue_load) for i in range(queue_num)]
        # self.queue = Queue.Queue(maxsize = queue_num)
        # for t in self.threads:
        #     t.setDaemon(True)
        #     t.start()

    def get(self):
        return self._load()
        # return self.queue.get()

    def get_video_num(self):
        return self.total_video_num

    # def _queue_load(self):
    #     while(True):
    #         data_labels = self._load()
    #         self.queue.put(data_labels)

    def _get_video_id(self):
        # self.mutex.acquire()
        ids = []
        for i in range(self.batch_size):
            ids.append(self.video_id)
            self.video_id += 1
            if self.video_id >= self.total_video_num:
                self.video_id = 0
        return ids
        # self.mutex.release()

    def _normal_load(self):
        data = np.zeros([self.batch_size, 224, 224, self.num_segments*self.num_length*img_channel], dtype=np.float32)
        labels = np.zeros(self.batch_size, dtype=np.int64)
    def _fulltest_load(self):
        data = np.zeros([self.batch_size*self.num_segments, 224, 224, self.num_length*img_channel], dtype=np.float32)
        labels = np.zeros(self.batch_size*self.num_segments, dtype=np.int64)

        video_block = video_block.reshape((224, 224, self.num_segments, self.num_length*img_channel))
        # to [num_segments, 224, 224, num_length*3/2]
        video_block = np.transpose(video_block, (2, 0, 1, 3))
        data[i*self.num_segments:(i+1)*self.num_segments, :, :, :] = video_block
        labels[i*self.num_segments:(i+1)*self.num_segments] = video_label
    def _seq_load(self):
        data, labels = self._load_batch()
        data = data.reshape([self.batch_size, 224, 224, self.num_segments, self.num_length*self.img_channel])
        data = np.transpose(data, [3, 0, 1, 2, 4])
        return data, labels

    def _load_batch(self):
        data = np.zeros([self.batch_size, 224, 224, self.num_segments*self.num_length*self.img_channel], dtype=np.float32)
        labels = np.zeros(self.batch_size, dtype=np.int64)
        video_ids = self._get_video_id()
        for i, vid in enumerate(video_ids):
            print "**** start loading batch", i
            video_item = self.video_list[vid]
            video_dir = video_item[0]
            video_length = video_item[1]
            video_label = video_item[2]
            print "**** before into _read_video_segments"
            data[i, :, :, :] = self._read_video_segments(video_dir, video_length)
            print "**** after into _read_video_segments"
            print "------------------------------------------"
            print
            labels[i] = video_label
        return data, labels

    def _read_video_segments(self, video_dir, video_length):
        print "in _read_video_segments"
        video_dir = os.path.join(self.root_dir, video_dir)
        frame_ids = self._get_equal_length_segments_start_id(video_length)
        # print("reading {}, {}".format(video_dir, frame_ids))
        images = np.zeros([256, 341, self.num_segments*self.num_length*self.img_channel], dtype=np.float32)
        start_time = time()
        print "start reading images", len(frame_ids)
        if not self.is_flow:
            for idx, frame_id in enumerate(frame_ids):
                img = cv2.imread(os.path.join(video_dir, '%04d.jpg'%frame_id))
                assert isinstance(img, np.ndarray)
                images[:, :, 3*idx:3*(idx+1)] = img
        else:
            for idx, frame_id in enumerate(frame_ids):
                img_x = cv2.imread(os.path.join(video_dir, '%04d_x.jpg'%frame_id), cv2.IMREAD_GRAYSCALE)
                img_y = cv2.imread(os.path.join(video_dir, '%04d_y.jpg'%frame_id), cv2.IMREAD_GRAYSCALE)
                assert isinstance(img_x, np.ndarray)
                assert isinstance(img_y, np.ndarray)
                images[:, :, 2*idx] = img_x
                images[:, :, 2*idx+1] = img_y
        read_time = time()
        print "start to do transformation"
        images = self.transformer.transform(images)
        trans_time = time()
        print "over in read seg: ", read_time - start_time, trans_time - read_time

        return images

    def _get_equal_length_segments_start_id(self, duration):
        avg_length = float(duration - self.num_length + 1) / float(self.num_segments)
        frame_ids = []
        if self.is_test:
            for seg_id in range(self.num_segments):
                seg_start_id = int(avg_length * (seg_id + 0.5))
                frame_ids += range(seg_start_id, seg_start_id+self.num_length)
        else:
            for seg_id in range(self.num_segments):
                seg_start_id = int(avg_length * (seg_id + random.random()))
                frame_ids += range(seg_start_id, seg_start_id+self.num_length)
        return frame_ids
