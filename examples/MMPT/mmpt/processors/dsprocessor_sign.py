"""
Processors for SignCLIP
"""

import json
import os
import pickle
import random
import math
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from .processor import (
    MetaProcessor,
    VideoProcessor,
    TextProcessor,
)
from .dsprocessor import DSAligner

# -------------------- SignCLIP common -----------------------

from pose_format import Pose
from pose_format.utils.normalization_3d import PoseNormalizer
from pose_format.utils.generic import pose_normalization_info
# from pose_format.utils.generic import flip_holistic, is_left_handed, pose_normalization_info

# import mediapipe as mp
# mp_holistic = mp.solutions.holistic
# FACEMESH_CONTOURS_POINTS = [str(p) for p in sorted(set(p for p_tup in mp_holistic.FACEMESH_CONTOURS for p in p_tup))]

# To avoid installing mediapipe, we just hardcode the face contours given the above code
FACEMESH_CONTOURS_POINTS = [
    '0', '7', '10', '13', '14', '17', '21', '33', '37', '39', '40', '46', '52', '53', '54', '55', '58', '61', '63',
    '65', '66', '67', '70', '78', '80', '81', '82', '84', '87', '88', '91', '93', '95', '103', '105', '107', '109',
    '127', '132', '133', '136', '144', '145', '146', '148', '149', '150', '152', '153', '154', '155', '157', '158',
    '159', '160', '161', '162', '163', '172', '173', '176', '178', '181', '185', '191', '234', '246', '249', '251',
    '263', '267', '269', '270', '276', '282', '283', '284', '285', '288', '291', '293', '295', '296', '297', '300',
    '308', '310', '311', '312', '314', '317', '318', '321', '323', '324', '332', '334', '336', '338', '356', '361',
    '362', '365', '373', '374', '375', '377', '378', '379', '380', '381', '382', '384', '385', '386', '387', '388',
    '389', '390', '397', '398', '400', '402', '405', '409', '415', '454', '466'
]

class PoseProcessor(VideoProcessor):
    def __init__(self, config):
        super().__init__(config)
        self.pose_components = config.pose_components
        self.normalize_hand = config.normalize_hand
        self.flip_pose = config.flip_pose
        self.augment2d = config.augment2d
        self.augment_temporal = config.augment_temporal
        self.augment_temporal_chance = config.augment_temporal_chance
        self.gaussian_noise = config.gaussian_noise
        self.max_video_len = config.max_video_len
        self.preprocess = config.preprocess
        self.anonym_pose = config.anonym_pose
        self.is_training = (config.split == 'train') and (not config.train_for_test)

        np.random.seed(42)

    def __call__(self, video_id, pose=None):
        if video_id:
            with open(os.path.join(self.vfeat_dir, video_id + ".pose"), "rb") as f:
                buffer = f.read()
                pose = Pose.read(buffer)

        # select components
        if self.pose_components:
            if self.pose_components == 'reduced_face':
                pose = pose.get_components(["POSE_LANDMARKS", "FACE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"], 
                    {"FACE_LANDMARKS": FACEMESH_CONTOURS_POINTS})
            else:
                pose = pose.get_components(self.pose_components)
                # 3D Hand Normalization
                if self.pose_components == ['RIGHT_HAND_LANDMARKS'] and self.normalize_hand:
                    pose = self.hand_normalization(pose)

        if self.flip_pose == 'right':
            # if is_left_handed(pose):
            if np.nan_to_num(pose.get_components(["RIGHT_HAND_LANDMARKS"]).body.data).var(axis=0).sum() == 0:
                pose = flip_holistic(pose)

        if self.anonym_pose:
            from pose_anonymization.appearance import remove_appearance
            # remove appearance + add spreadthesign mean pose
            # https://github.com/sign-language-processing/pose-anonymization
            pose = remove_appearance(pose)

        if self.preprocess == 'sign-vq' or self.preprocess == 'sign-vq-original-scale':
            from sign_vq.data.normalize import pre_process_mediapipe, normalize_mean_std
            # reuse the preprocessing pipeline from sign-vq
            # https://github.com/sign-language-processing/sign-vq
            pose = pre_process_mediapipe(pose)

            if not self.preprocess == 'sign-vq-original-scale':
                # this removes spreadthesign mean pose
                pose = normalize_mean_std(pose)
        else:
            # normalize pose: the mean distance between the shoulders of each person equals 1
            pose = pose.normalize(self.pose_normalization_info(pose.header))
            pose = self.pose_hide_legs(pose)

        # augmentation (training only)
        if self.is_training:
            if self.flip_pose and random.random() < self.flip_pose:
                # TODO: move the flip_pose function into pose-format

                # CAUTION: flipping works on reduced set of key points only
                FLIPPED_COMPONENTS = ["POSE_LANDMARKS", "FACE_LANDMARKS", "RIGHT_HAND_LANDMARKS", "LEFT_HAND_LANDMARKS"]
                # FLIPPED_BODY_POINTS = ['RIGHT_SHOULDER', 'LEFT_SHOULDER', 'RIGHT_ELBOW', 'LEFT_ELBOW', 'RIGHT_WRIST', 'LEFT_WRIST', 'RIGHT_HIP', 'LEFT_HIP']
                FLIPPED_BODY_POINTS = ['NOSE', 'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EAR', 'LEFT_EAR', 'MOUTH_RIGHT', 'MOUTH_LEFT', 'RIGHT_SHOULDER', 'LEFT_SHOULDER', 'RIGHT_ELBOW', 'LEFT_ELBOW', 'RIGHT_WRIST', 'LEFT_WRIST', 'RIGHT_PINKY', 'LEFT_PINKY', 'RIGHT_INDEX', 'LEFT_INDEX', 'RIGHT_THUMB', 'LEFT_THUMB', 'RIGHT_HIP', 'LEFT_HIP', 'RIGHT_KNEE', 'LEFT_KNEE', 'RIGHT_ANKLE', 'LEFT_ANKLE', 'RIGHT_HEEL', 'LEFT_HEEL', 'RIGHT_FOOT_INDEX', 'LEFT_FOOT_INDEX']
                # face flipping based on https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png
                FLIPPED_FACE_POINTS = ['0', '249', '10', '13', '14', '17', '251', '263', '267', '269', '270', '276', '282', '283', '284', '285', '288', '291', '293', '295', '296', '297', '300', '308', '310', '311', '312', '314', '317', '318', '321', '323', '324', '332', '334', '336', '338', '356', '361', '362', '365', '373', '374', '375', '377', '378', '379', '152', '380', '381', '382', '384', '385', '386', '387', '388', '389', '390', '397', '398', '400', '402', '405', '409', '415', '454', '466', \
                                            '7', '21', '33', '37', '39', '40', '46', '52', '53', '54', '55', '58', '61', '63', '65', '66', '67', '70', '78', '80', '81', '82', '84', '87', '88', '91', '93', '95', '103', '105', '107', '109', '127', '132', '133', '136', '144', '145', '146', '148', '149', '150', '153', '154', '155', '157', '158', '159', '160', '161', '162', '163', '172', '173', '176', '178', '181', '185', '191', '234', '246']
                pose = pose.flip(0).get_components(FLIPPED_COMPONENTS, {"POSE_LANDMARKS": FLIPPED_BODY_POINTS, "FACE_LANDMARKS": FLIPPED_FACE_POINTS})
                pose = pose.normalize(self.pose_normalization_info(pose.header))
            if self.augment2d:
                pose = pose.augment2d()
            if self.augment_temporal and pose.body.data.shape[0] > 1:
                if not self.augment_temporal_chance or random.random() < self.augment_temporal_chance:
                    old_fps = pose.body.fps
                    ratio = np.random.normal(loc=1, scale=0.2)
                    if pose.body.data.shape[0] * ratio < self.max_video_len:
                        new_fps = round(old_fps * ratio)
                        pose = pose.interpolate(new_fps, kind='linear')
            if self.gaussian_noise:
                noise_std = 0.001
                noise = np.random.normal(scale=noise_std, size=pose.body.data.shape)
                pose.body.data = pose.body.data + noise

        feat = np.nan_to_num(pose.body.data)
        feat = feat.reshape(feat.shape[0], -1)
        
        return feat

    def pose_normalization_info(self, pose_header):
        if pose_header.components[0].name == "POSE_LANDMARKS":
            return pose_header.normalization_info(p1=("POSE_LANDMARKS", "RIGHT_SHOULDER"),
                                                p2=("POSE_LANDMARKS", "LEFT_SHOULDER"))

        if pose_header.components[0].name == "BODY_135":
            return pose_header.normalization_info(p1=("BODY_135", "RShoulder"), p2=("BODY_135", "LShoulder"))

        if pose_header.components[0].name == "pose_keypoints_2d":
            return pose_header.normalization_info(p1=("pose_keypoints_2d", "RShoulder"),
                                                p2=("pose_keypoints_2d", "LShoulder"))

        raise ValueError("Unknown pose header schema for normalization")

    def hand_normalization(self, pose):
        plane = pose.header.normalization_info(
            p1=("RIGHT_HAND_LANDMARKS", "WRIST"),
            p2=("RIGHT_HAND_LANDMARKS", "PINKY_MCP"),
            p3=("RIGHT_HAND_LANDMARKS", "INDEX_FINGER_MCP")
        )
        line = pose.header.normalization_info(
            p1=("RIGHT_HAND_LANDMARKS", "WRIST"),
            p2=("RIGHT_HAND_LANDMARKS", "MIDDLE_FINGER_MCP")
        )
        normalizer = PoseNormalizer(plane=plane, line=line, size=100)
        tensor = normalizer(pose.body.data)

        pose.body.data = tensor
        pose.focus()

        return pose

    def pose_hide_legs(self, pose):
        if pose.header.components[0].name == "POSE_LANDMARKS":
            point_names = ["KNEE", "ANKLE", "HEEL", "FOOT_INDEX"]
            # pylint: disable=protected-access
            points = [
                pose.header._get_point_index("POSE_LANDMARKS", side + "_" + n)
                for n in point_names
                for side in ["LEFT", "RIGHT"]
            ]
            pose.body.confidence[:, :, points] = 0
            pose.body.data[:, :, points, :] = 0
            return pose
        else:
            raise ValueError("Unknown pose header schema for hiding legs")


# -------------------- RWTH Fingerspelling -----------------------


class RWTHFSMetaProcessor(MetaProcessor):
    """RWTH German Fingerspelling Database
    https://www-i6.informatik.rwth-aachen.de/aslr/fingerspelling.php
    """

    def __init__(self, config):
        super().__init__(config)

        vfeat_dir = config.vfeat_dir
        split_path = self._get_split_path(config)

        self.letter_to_id = {}
        self.id_to_letter = {}

        with open(config.gesture_id_path) as f:
            for idx, line in enumerate(f):
                letter = line.split(' = ')[1].rstrip('\n')
                self.letter_to_id[letter] = idx + 1
                self.id_to_letter[str(idx + 1)] = letter

        with open(split_path) as f:
            lines = []
            for line in f:
                video_id = line.rstrip('\n') 
                signer_id, letter_id, seq_id, camera_id = video_id.split('_')

                # FIXME: for now we do full body pose estimation for all videos, so exclude cam1 where only the hands are present
                if config.video_processor == 'RWTHFSPoseProcessor' and camera_id == 'cam1':
                    continue
                
                lines.append(video_id)

            if config.split == 'train':
                self.data = []

                video_ids = defaultdict(list)
                for video_id in lines:
                    signer_id, letter_id, seq_id, camera_id = video_id.split('_')
                    video_ids[self.id_to_letter[letter_id]].append(video_id)

                length = []
                for key, value in video_ids.items():
                    length.append(len(value))
                max_length = max(length)

                for i in range(max_length):
                    for key, value in video_ids.items():
                        self.data.append(value[i % len(value)])
            else:
                self.data = lines

    def __getitem__(self, idx):
        video_id = self.data[idx]
        signer_id, letter_id, seq_id, camera_id = video_id.split('_')
        body_part = 'handshape' if camera_id == 'cam1' else 'whole body'
        text_info = f'Fingerspell the letter {self.id_to_letter[letter_id]} in German Sign Language.'
        # print(video_id, text_info)
        return video_id, text_info


class RWTHFSVideoProcessor(VideoProcessor):
    def __call__(self, video_id):
        feat = np.load(os.path.join(self.vfeat_dir, video_id + ".npy"))
        # pooling adapater (not needed when training from scratch)
        feat_dim = 512
        if feat.shape[1] > feat_dim and not self.vfeat_custom:
            # i3d feature is 1024
            # adapt feature dimension to 512 by average pooling
            feat = feat.reshape(feat.shape[0], feat_dim, int(feat.shape[1] / feat_dim))
            feat = np.average(feat, axis=2)
        return feat


class RWTHFSPoseProcessor(PoseProcessor):
    pass

# -------------------- ASL Signs -----------------------

class ASLSignMetaProcessor(MetaProcessor):
    """Google - Isolated Sign Language Recognition
    https://www.kaggle.com/competitions/asl-signs/overview
    """

    def __init__(self, config):
        super().__init__(config)

        vfeat_dir = config.vfeat_dir
        split_path = self._get_split_path(config)
        metadata_df = pd.read_csv(config.metadata_path, dtype=str)

        with open(split_path) as f:
            lines = []
            for line in f:
                video_id = line.rstrip('\n') 
                lines.append(video_id)

            metadata_df = metadata_df[metadata_df['sequence_id'].isin(lines)]
            data = metadata_df.to_dict('records')

            print(f'sign distribution in the {config.split} set:')
            print(metadata_df.groupby(['sign'])['sign'].count().reset_index(name='count').sort_values(['count'], ascending=False))

            if config.split == 'train':
                self.data = []

                indices = defaultdict(list)
                for index, item in enumerate(data):
                    indices[item['sign']].append(index)

                length = []
                for key, value in indices.items():
                    length.append(len(value))
                max_length = max(length)

                for i in range(max_length):
                    for key, value in indices.items():
                        self.data.append(data[value[i % len(value)]])
            else:
                self.data = data

    def __getitem__(self, idx):
        video_id = self.data[idx]['path'].replace('train_landmark_files/', '')
        text_info = f'Sign the sign "{self.data[idx]["sign"]}" in American Sign Language.'
        # print(video_id, text_info)
        return video_id, text_info


class ASLSignPoseProcessor(PoseProcessor):
    NOSE = [
        1,2,98,327
    ]
    LIP = [ 0, 
        61, 185, 40, 39, 37, 267, 269, 270, 409,
        291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
        95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    ]
    REYE = [
        33, 7, 163, 144, 145, 153, 154, 155, 133,
        246, 161, 160, 159, 158, 157, 173,
    ]
    LEYE = [
        263, 249, 390, 373, 374, 380, 381, 382, 362,
        466, 388, 387, 386, 385, 384, 398,
    ]
    FACE = sorted(NOSE + LIP + REYE + LEYE)
    FACE_FULL = np.arange(0, 468).tolist()

    LHAND = np.arange(468, 489).tolist()
    POSE = np.arange(489, 522).tolist()
    RHAND = np.arange(522, 543).tolist()
    BODY = LHAND + POSE + RHAND

    def __call__(self, video_id):
        import pyarrow.parquet as pq

        pose_df = pq.read_table(os.path.join(self.vfeat_dir, video_id)).to_pandas()

        # pd.set_option('display.max_rows', None)
        # print(pose_df[pose_df['frame'] == 18])

        # pose_df = pose_df[pose_df['type'].isin(self.pose_components)]

        points = []
        if "face" in self.pose_components:
            points = points + self.FACE
        if "face_full" in self.pose_components:
            points = points + self.FACE_FULL
        if "left_hand" in self.pose_components:
            points = points + self.LHAND
        if "pose" in self.pose_components:
            points = points + self.POSE    
        if "right_hand" in self.pose_components:
            points = points + self.RHAND

        num_frames = len(pose_df['frame'].drop_duplicates())
        dimensions = ['x', 'y', 'z']

        pose_data = pose_df[dimensions].to_numpy().reshape(num_frames, -1, len(dimensions))
        pose_data = pose_data[:, points, :]

        pose_data = pose_data.reshape(num_frames, -1)
        pose_data = np.nan_to_num(pose_data)
        
        return pose_data


# -------------------- SignCLIP pretrained on Spreadthesign -----------------------

import re
import string
import importlib
from tqdm import tqdm

from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.pose_header import PoseHeader
from pose_format.utils.reader import BufferReader
        

class SignCLIPPretrainMetaProcessor(MetaProcessor):
    def __init__(self, config):
        super().__init__(config)
        random.seed(42)

        if config.debug:
            config.split = 'test'

        self.vfeat_dir = config.vfeat_dir
        self.task = config.task
        split_path = self._get_split_path(config)
        metadata_df = pd.read_csv(config.metadata_path, dtype=str)

        with open(split_path) as f:
            lines = []
            for line in f:
                video_id = int(line.rstrip('\n'))
                lines.append(video_id)

            metadata_df = metadata_df.reset_index()
            metadata_df = metadata_df[metadata_df['index'].isin(lines)]
            metadata_df = metadata_df[metadata_df['language'] == 'en']

            print(metadata_df)

            print(f'text distribution in the {config.split} set:')
            print(metadata_df.groupby(['text'])['text'].count().reset_index(name='count').sort_values(['count'], ascending=False))

            print(f'language distribution in the {config.split} set:')
            print(metadata_df.groupby(['videoLanguage'])['videoLanguage'].count().reset_index(name='count').sort_values(['count'], ascending=False))

            data = metadata_df.to_dict('records')
            data = [datum for datum in tqdm(data) if os.path.exists(os.path.join(config.vfeat_dir, datum['pose']))]

            print(f'In total {len(data)} {config.split} examples with poses.')

            self.data = data

            if config.split == 'train':
                random.shuffle(self.data)

    def __getitem__(self, idx):
        datum = self.data[idx]
        video_id = datum['pose'].replace('.pose', '')
        text = '' if self.task == 'identification' else datum['text']
        vlan = '<ase>' if self.task == 'conceptualization' else f"<{datum['videoLanguage']}>"
        text_info = f"<{datum['language']}> {vlan} {text}"
        return video_id, text_info


# -------------------- SignCLIP v1 (ASL, BSL, etc.) (feat. sign_language_datasets) -----------------------

class SignCLIPPoseProcessor(PoseProcessor):
    def __call__(self, pose):
        # the pose objects are passed to PoseProcessor
        feat = super().__call__(None, pose)
        return feat


class SignCLIPMetaProcessor(MetaProcessor):
    def __init__(self, config):
        super().__init__(config)
        random.seed(42)

        import tensorflow_datasets as tfds
        import sign_language_datasets.datasets
        from sign_language_datasets.datasets.config import SignDatasetConfig

        self.config = config
        self.task = config.task
        self.split = config.split
        self.pose_processer = SignCLIPPoseProcessor(config) # call pose_processer by meta_processor itself
        self.datasets = {}
        self.data = []

        if config.test_in_vocab:
            vocab_path = './data_stat_sp_concept_dis.csv'
            vocab_df = pd.read_csv(vocab_path)
            vocab_df = vocab_df[vocab_df['count'] > 20]
            self.vocab = list(vocab_df['text'])

        print('================================')
        print(f'Loading {self.split} data ... ')
        print('================================')

        datasets = config[f'{"test" if config.train_for_test else self.split}_datasets']
        datasets = [item if len(item) == 3 else [*item, None] for item in datasets]

        for dataset, version, split_version in datasets:
            print('--------------------------------')
            print(f'Loading the {dataset} {version} dataset, {split_version if split_version else "default"} split ... ')
            print('--------------------------------')

            # read common pose header for the dataset
            dataset_module = importlib.import_module("sign_language_datasets.datasets." + dataset + "." + dataset)
            with open(dataset_module._POSE_HEADERS['holistic'], "rb") as buffer:
                pose_header = PoseHeader.read(BufferReader(buffer.read()))

            sd_config = SignDatasetConfig(name=config.config_name or 'holistic', version=version, include_video=False, include_pose="holistic", extra={'split': split_version} if split_version else {})
            splits = ['validation' if self.split == 'valid' else self.split]
            # utilize unused validation data from some datasets for pretraining as well
            if self.split == 'train' and config.use_valid_for_pretraining and dataset not in [d[0] for d in config.valid_datasets]:
                splits = ['train', 'validation'] 

            for split in splits:
                data_l = tfds.load(name=dataset, builder_kwargs=dict(config=sd_config), data_dir=config.data_dir)[split]

                print(f'In total {len(data_l)} raw {split} examples.')
                print('Iterate over examples ...')

                self.datasets[dataset] = {
                    'pose_header': pose_header,
                    'data_l': data_l,
                }

                count = 0
                for dataset_index, datum in enumerate(tqdm(data_l)):
                    if not (datum['pose']['data'].shape[0] > 0 and datum['pose']['data'].shape[0] <= config.max_video_len):
                        continue

                    if config.debug and count >= 1000:
                        break
                    count = count + 1

                    if config.sp_universal_tagging:
                        # HACK: fine-tuning only for ase and bfi at the moment
                        if dataset == 'bobsl_islr':
                            tag_prompt = "<en> <bfi>" 
                        else:
                            tag_prompt = "<en> <ase>" 
                    else:
                        tag_prompt = "<American Sign Language>"

                    text_content = datum['text'].numpy().decode('utf-8')

                    if config.test_in_vocab or config.preprocess_gloss:
                        if dataset == 'asl_citizen':
                            text_content = text_content.lower()
                            text_content = text_content.rstrip(string.digits)
                        elif dataset == 'sem_lex':
                            # text_content = text_content.rstrip(string.digits)
                            # text_content = text_content.rstrip('_')
                            text_content = re.sub(r'_\d+$', '', text_content)
                            text_content = text_content.replace('_', ' ')

                    if config.test_in_vocab and text_content not in self.vocab:
                        continue

                    text_prompt = f"{tag_prompt} {text_content}"

                    # save memory consumption for 3.5M BOBSL ISLR examples to under 300GB
                    # better solution: use SignCLIPMetaProcessorV2 to load data asynchronously
                    if config.pre_compute_vfeat: 
                        # reconstruct pose object
                        tf_pose = datum['pose']
                        fps = int(tf_pose["fps"].numpy())
                        pose_body = NumPyPoseBody(fps, tf_pose["data"].numpy(), tf_pose["conf"].numpy())
                        pose = Pose(pose_header, pose_body)
                        vfeat = self.pose_processer(pose)

                        self.data.append(dict(
                            id=f"{dataset}_{datum['id'].numpy().decode('utf-8')}",
                            text=text_prompt,
                            vfeat=vfeat,
                        ))
                    else:
                        self.data.append(dict(
                            datum,
                            dataset=dataset,
                            id=f"{dataset}_{datum['id'].numpy().decode('utf-8')}",
                            text=text_prompt,
                        ))

                print(f'In total {count} wellformed {split} examples.')

        print(f'In total {len(self.data)} wellformed {split} examples from all datasets.')
        
        if self.split == 'train':
            random.shuffle(self.data)

        # Group examples by text prompts
        self.text_to_idxs = defaultdict(list)
        for idx, datum in enumerate(self.data):
            self.text_to_idxs[datum['text']].append(idx)

        print('Number of examples grouped by the text prompts:')
        text_to_idxs_num = [(text, len(idxs)) for text, idxs in self.text_to_idxs.items()]
        text_to_idxs_num = sorted(text_to_idxs_num, key=lambda x: x[1], reverse=True)
        for i, entry in enumerate(text_to_idxs_num):
            if i < 10 or (len(text_to_idxs_num) - i < 10):
                print(entry)
            elif i == 10:
                print('...')
            # print(entry)
        print('Total classes:', len(text_to_idxs_num))
        
    def __getitem__(self, idx):
        datum = self.data[idx]

        # vfeat pre-computed during __init__
        if 'vfeat' in datum:
            return idx, datum['text'], datum['vfeat']

        # reconstruct pose object
        tf_pose = datum['pose']
        fps = int(tf_pose["fps"].numpy())
        pose_body = NumPyPoseBody(fps, tf_pose["data"].numpy(), tf_pose["conf"].numpy())
        dataset = self.datasets[datum['dataset']]
        pose = Pose(dataset['pose_header'], pose_body)
        vfeat = self.pose_processer(pose)

        return idx, datum['text'], vfeat


class SignCLIPMetaProcessorV2(MetaProcessor):
    def __init__(self, config):
        super().__init__(config)
        random.seed(42)

        import tensorflow_datasets as tfds
        import sign_language_datasets.datasets
        from sign_language_datasets.datasets.config import SignDatasetConfig

        self.config = config
        self.split = config.split
        self.pose_processer = SignCLIPPoseProcessor(config) # call pose_processer by meta_processor itself

        print('================================')
        print(f'Loading {self.split} data ... ')
        print('================================')

        datasets = config[f'{"test" if config.train_for_test else self.split}_datasets']
        datasets = [item if len(item) == 3 else [*item, None] for item in datasets]

        for dataset, version, split_version in datasets:
            print('--------------------------------')
            print(f'Loading the {dataset} {version} dataset, {split_version if split_version else "default"} split ... ')
            print('--------------------------------')

            # read common pose header for the dataset
            dataset_module = importlib.import_module("sign_language_datasets.datasets." + dataset + "." + dataset)
            with open(dataset_module._POSE_HEADERS['holistic'], "rb") as buffer:
                pose_header = PoseHeader.read(BufferReader(buffer.read()))

            sd_config = SignDatasetConfig(name=config.config_name or 'holistic', version=version, include_video=False, include_pose="holistic", extra={
                'split': split_version,
                'lip_feature_dir': config.lip_feature_dir if config.config_name == 'holistic_lip' else None,
            })
            split = 'validation' if self.split == 'valid' else self.split

            self.pose_header = pose_header

            if config.debug:
                split_debug = 'validation' if split == 'validation' else 'test'
                self.data_l = list(tfds.load(name=dataset, builder_kwargs=dict(config=sd_config), data_dir=config.data_dir)[split_debug])[:1000]

                # Group examples by text prompts
                self.text_to_idxs = defaultdict(list)
                for idx, datum in enumerate(self.data_l):
                    self.text_to_idxs[datum['text'].numpy().decode('utf-8')].append(idx)

                print('Number of examples grouped by the text prompts:')
                text_to_idxs_num = [(text, len(idxs)) for text, idxs in self.text_to_idxs.items()]
                text_to_idxs_num = sorted(text_to_idxs_num, key=lambda x: x[1], reverse=True)
                for i, entry in enumerate(text_to_idxs_num):
                    if i < 10 or (len(text_to_idxs_num) - i < 10):
                        print(entry)
                    elif i == 10:
                        print('...')
                print('Total classes:', len(text_to_idxs_num))
            else:
                self.data_l = tfds.load(name=dataset, builder_kwargs=dict(config=sd_config), data_dir=config.data_dir)[split]

            print("Dataset initialized")
            print(f'In total {len(self.data_l)} raw {split} examples.')


    def reset_data_iter(self):
        print("Resetting iterator")
        self.data_iter = iter(self.data_l)


    def __len__(self): 
        return len(self.data_l)


    def __getitem__(self, idx):
        # print(f"Fetching {self.split} item {idx}")

        if idx == 0:
            self.reset_data_iter()
            self.current_idx = 0
        else:
            assert (self.current_idx + 1) == idx, "assumes sequential access items"
            self.current_idx = idx

        try:
            # Get the next data from the TensorFlow generator
            datum = next(self.data_iter)
        except StopIteration:
            print("Iterator exhausted, resetting...")
            # If the generator is exhausted, reset it
            self.reset_data_iter()
            # Retrieve the first element from the reset generator
            datum = next(self.data_iter)
            self.current_idx = 0

        example_id = datum['id'].numpy().decode('utf-8')

        tag_prompt = "<en> <bfi>" # FIXME
        text_content = datum['text'].numpy().decode('utf-8')
        text_prompt = f"{tag_prompt} {text_content}"

        if self.config.only_lip_reading:
            vfeat = datum['lip'].numpy()
        else:
            # reconstruct pose object
            tf_pose = datum['pose']
            fps = int(tf_pose["fps"].numpy())
            pose_body = NumPyPoseBody(fps, tf_pose["data"].numpy(), tf_pose["conf"].numpy())
            pose = Pose(self.pose_header, pose_body)
            vfeat = self.pose_processer(pose)

            if self.config.include_lip_reading:
                lip_feat = datum['lip'].numpy()
                vfeat = np.concatenate((vfeat, lip_feat), axis=1)

        return example_id, text_prompt, vfeat


# -------------------- SignCLIP-Suisse -----------------------


class SignCLIPSuisseMetaProcessor(MetaProcessor):
    def __init__(self, config):
        super().__init__(config)
        random.seed(42)

        self.pose_processer = SignCLIPPoseProcessor(config) # call pose_processer by meta_processor itself
        split = 'val' if config.split == 'valid' else config.split
        metadata_df = pd.read_csv(config[f'{split}_path'], dtype=str)

        print(f'language distribution in the {config.split} set:')
        print(metadata_df.groupby(['signedLanguage'])['signedLanguage'].count().reset_index(name='count').sort_values(['count'], ascending=False))

        data = metadata_df.to_dict('records')
        self.data = []
        language_map = {
            'dsgs': 'sgg',
            'lsf-ch': 'fsl',
            'lis-ch': 'ise',
        }

        if config.debug:
            data = data[:10]

        for datum in tqdm(data):
            spoken_lan = datum['spokenLanguage']
            sign_lan = language_map[datum['signedLanguage']]

            pose_path = os.path.join(config.vfeat_dir, datum['id'] + ".pose")
            with open(pose_path, "rb") as f:
                text = datum['name']
                # text = re.sub(r' \d+$', '', text)
                # text = text.lower().title()
                self.data.append({
                    'text': f"<{spoken_lan}> <{sign_lan}> {text}",
                    'pose': Pose.read(f),
                })

            example_pose_path = os.path.join(config.vfeat_example_dir, datum['id'] + ".pose")
            if os.path.exists(example_pose_path):
                with open(example_pose_path, "rb") as f:
                    text = datum['example']
                    self.data.append({
                        'text': f"<{spoken_lan}> <{sign_lan}> {text}",
                        'pose': Pose.read(f),
                    })

        print(f'In total {len(self.data)} {config.split} examples with poses.')

        if config.split == 'train':
            random.shuffle(self.data)

        print('Print some example text prompts:')
        for datum in self.data[:20]:
            print(datum['text'])

    def __getitem__(self, idx):
        datum = self.data[idx]
        vfeat = self.pose_processer(datum['pose'])


# -------------------- SignCLIP CNN (appearance-based) -----------------------
# Appearance-based variant of SignCLIP that uses pre-extracted CNN features
# (e.g. I3D BSL-1K CVPR'21 136MB) instead of MediaPipe pose features.
#
# Workflow:
#   1. Extract features offline with extract_asl_citizen_i3d_features.py
#      → saves {dataset}_{video_id}.npy files to a flat directory
#   2. Use this MetaProcessor + RWTHFSVideoProcessor in the config:
#        meta_processor: SignCLIPVideoMetaProcessor
#        video_processor: RWTHFSVideoProcessor
#        vfeat_dir: /path/to/i3d_features
#        vfeat_custom: 1          # keep full 1024-dim (no 512 pooling)
#        vfeat_dim: 1024          # (under model:)
#   3. Start from a modified E7.2 checkpoint (prepare_cnn_checkpoint.py)
#      to keep 12 video BERT layers and the text encoder.
#
# Returns a 2-tuple (feat_id, text_prompt) so that mmdataset.py routes
# through video_processor(feat_id) for .npy loading (standard path).


class SignCLIPVideoMetaProcessor(MetaProcessor):
    """MetaProcessor for appearance-based SignCLIP.

    Uses tensorflow_datasets / sign_language_datasets for train/val/test splits
    and text labels, but loads pre-extracted CNN features from .npy files
    instead of computing pose features.  The .npy files must be named:
        {dataset}_{datum_id}.npy
    and placed in config.vfeat_dir.
    """

    def __init__(self, config):
        super().__init__(config)
        random.seed(42)

        import tensorflow_datasets as tfds
        import sign_language_datasets.datasets  # noqa: F401
        from sign_language_datasets.datasets.config import SignDatasetConfig

        self.vfeat_dir = config.vfeat_dir
        self.split = config.split
        self.data = []

        print('================================')
        print(f'Loading CNN feature metadata ({self.split}) ...')
        print('================================')

        datasets = config[f'{"test" if config.train_for_test else self.split}_datasets']
        datasets = [item if len(item) == 3 else [*item, None] for item in datasets]

        for dataset, version, split_version in datasets:
            print('--------------------------------')
            print(f'Loading {dataset} {version}, split {split_version} ...')
            print('--------------------------------')

            # Use same config name as the existing pose-based setup so that the
            # already-built tfds cache on the cluster is reused.
            sd_config = SignDatasetConfig(
                name=config.config_name or 'holistic',
                version=version,
                include_video=False,
                include_pose="holistic",
                extra={'split': split_version} if split_version else {},
            )
            split = 'validation' if self.split == 'valid' else self.split
            data_l = tfds.load(
                name=dataset,
                builder_kwargs=dict(config=sd_config),
                data_dir=config.data_dir,
            )[split]

            count = 0
            missing = 0
            for datum in tqdm(data_l):
                datum_id = datum['id'].numpy().decode('utf-8')
                text_content = datum['text'].numpy().decode('utf-8')
                feat_id = f"{dataset}_{datum_id}"
                feat_path = os.path.join(self.vfeat_dir, feat_id + ".npy")

                if not os.path.exists(feat_path):
                    missing += 1
                    continue

                if config.test_in_vocab or config.preprocess_gloss:
                    if dataset == 'asl_citizen':
                        text_content = text_content.lower()
                        text_content = text_content.rstrip(string.digits)
                    elif dataset == 'sem_lex':
                        text_content = re.sub(r'_\d+$', '', text_content)
                        text_content = text_content.replace('_', ' ')

                if config.sp_universal_tagging:
                    tag_prompt = "<en> <ase>"
                else:
                    tag_prompt = "<American Sign Language>"

                self.data.append({
                    'id': feat_id,
                    'text': f"{tag_prompt} {text_content}",
                })
                count += 1

            print(f'{dataset}: {count} examples with CNN features, {missing} missing.')

        if self.split == 'train':
            random.shuffle(self.data)

        self.text_to_idxs = defaultdict(list)
        for idx, datum in enumerate(self.data):
            self.text_to_idxs[datum['text']].append(idx)

        print(f'Total: {len(self.data)} examples, {len(self.text_to_idxs)} unique prompts.')

    def __getitem__(self, idx):
        datum = self.data[idx]
        # Return 2-tuple so mmdataset.py calls video_processor(feat_id) for .npy loading
        return datum['id'], datum['text']


# -------------------- SignCLIP CNN — CSV-based (no tfds) -----------------------
# Replaces SignCLIPVideoMetaProcessor for clusters where the tfds cache is not
# available.  Reads directly from the ASL-Citizen split CSVs:
#     {splits_dir}/train.csv  |  val.csv  |  test.csv
# Each CSV has columns: Participant ID, Video file, Gloss, ASL-LEX Code
#
# Config fields required:
#   splits_dir:   /path/to/ASL_Citizen/splits
#   vfeat_dir:    /path/to/ASL_Citizen/i3d_features
#   dataset_name: asl_citizen  (optional, default)


class SignCLIPVideoCSVMetaProcessor(MetaProcessor):
    """CSV-based MetaProcessor for appearance-based SignCLIP on ASL-Citizen.

    Does not require tensorflow_datasets.  Reads split CSVs shipped with the
    ASL-Citizen dataset.  Feature files must be named:
        {dataset_name}_{video_basename_without_ext}.npy
    """

    _SPLIT_FILES = {'train': 'train.csv', 'valid': 'val.csv', 'test': 'test.csv'}

    def __init__(self, config):
        super().__init__(config)
        random.seed(42)
        import csv

        self.vfeat_dir = config.vfeat_dir
        dataset_name = config.dataset_name or 'asl_citizen'

        split_key = 'test' if config.train_for_test else self.split
        csv_path = os.path.join(config.splits_dir, self._SPLIT_FILES[split_key])

        print(f'Loading CNN feature metadata ({split_key}) from {csv_path} ...')

        self.data = []
        count = missing = 0

        with open(csv_path, newline='') as f:
            for row in csv.DictReader(f):
                video_id = os.path.splitext(row['Video file'])[0]
                feat_id = f"{dataset_name}_{video_id}"
                feat_path = os.path.join(self.vfeat_dir, feat_id + '.npy')

                if not os.path.exists(feat_path):
                    missing += 1
                    continue

                tag = '<en> <ase>' if config.sp_universal_tagging else '<American Sign Language>'
                self.data.append({'id': feat_id, 'text': f"{tag} {row['Gloss']}"})
                count += 1

        if self.split == 'train':
            random.shuffle(self.data)

        self.text_to_idxs = defaultdict(list)
        for idx, datum in enumerate(self.data):
            self.text_to_idxs[datum['text']].append(idx)

        print(f'Loaded {count} examples ({missing} missing .npy), '
              f'{len(self.text_to_idxs)} unique glosses.')

    def __getitem__(self, idx):
        datum = self.data[idx]
        return datum['id'], datum['text']


# -------------------- NGT appearance-invariant contrastive data -----------------------
# Paired NGT data: each sign has a real Logos .npy (Bushuis) and an Unreal Engine
# Logos .npy (NGT_Aug palmer), linked by a JSON manifest produced by
# make_ngt_pair_manifest.py.  No gloss labels required.
#
# Config fields required:
#   pair_manifest:  /path/to/ngt_pair_manifest.json
#   vfeat_dir:      ignored (paths are absolute in the manifest)
#
# Usage in a .yaml config:
#   meta_processor:   NGTPairMetaProcessor
#   video_processor:  NGTPairVideoProcessor
#   aligner:          NGTPairAligner


class NGTPairMetaProcessor(MetaProcessor):
    """Yields (sign_id, dummy_text) for each sign in the NGT pair manifest.

    The dummy text is an empty string; the aligner will build a minimal
    [CLS][SEP] token sequence so the model's text side receives valid input
    even though no gloss label is available.
    """

    def __init__(self, config):
        super().__init__(config)
        with open(config.pair_manifest) as f:
            self.manifest = json.load(f)
        self.sign_ids = sorted(self.manifest.keys())
        if getattr(config, 'split', 'train') == 'train':
            random.shuffle(self.sign_ids)
        print(f'NGTPairMetaProcessor: {len(self.sign_ids)} paired signs loaded.')

    def __len__(self):
        return len(self.sign_ids)

    def __getitem__(self, idx):
        return self.sign_ids[idx], ''  # (sign_id, dummy_text)


class NGTPairVideoProcessor(VideoProcessor):
    """Loads N view features for a given sign_id from the pair manifest.

    Returns a tuple of N numpy arrays, each of shape (N_clips, 768).

    Manifest entry formats (both supported):
      - dict:  {"real": [...], "unreal": [...]}          (single recording)
      - list:  [{"real": [...], "unreal": [...]}, ...]   (one dict per repeat
        take of the same sentence; produced by make_ngt_singleview_manifest.py)
    For list entries, ONE take is sampled uniformly per __getitem__ during
    training, so every repeat take participates over training while a batch
    never contains two items of the same sentence (no false negatives).
    Outside training (split != 'train'), take 0 is used deterministically.

    Config flags:
      load_real (bool, default True):
          Include the real views listed under "real".
      load_unreal (bool, default True):
          Include synthetic views listed under "unreal".
      unreal_character (int or null, default null):
          Which synthetic character to load.  The full pair manifest stores
          unreal views as [palmer_cam2, palmer_cam3, palmer_cam4, digits_cam2,
          digits_cam3, digits_cam4], so 0 = palmer (views 0-2), 1 = digits
          (views 3-5), null = all views.

    Typical K values:
      load_real=true,  load_unreal=false                → K=3  (real baseline)
      load_real=false, load_unreal=true, character=0    → K=3  (synthetic baseline)
      load_real=true,  load_unreal=true                 → K=9  (intervention)
    """

    _VIEWS_PER_CHARACTER = 3

    def __init__(self, config):
        super().__init__(config)
        with open(config.pair_manifest) as f:
            self.manifest = json.load(f)
        _lr = getattr(config, 'load_real', None)
        self.load_real = True if _lr is None else bool(_lr)
        _lu = getattr(config, 'load_unreal', None)
        self.load_unreal = True if _lu is None else bool(_lu)
        _uc = getattr(config, 'unreal_character', None)
        self.unreal_character = None if (_uc is None or _uc == 'null') else int(_uc)
        self.split = getattr(config, 'split', 'train')

    def __call__(self, sign_id):
        paths = self.manifest[sign_id]
        if isinstance(paths, list):
            if self.split == 'train':
                paths = random.choice(paths)
            else:
                paths = paths[0]
        feats = []
        if self.load_real:
            feats += [np.load(p) for p in paths['real']]
        if self.load_unreal:
            unreal = paths.get('unreal', [])
            if self.unreal_character is not None:
                start = self.unreal_character * self._VIEWS_PER_CHARACTER
                unreal = unreal[start:start + self._VIEWS_PER_CHARACTER]
            feats += [np.load(p) for p in unreal]
        if not feats:
            raise RuntimeError(
                f"[NGTPairVideoProcessor] No features loaded for sign '{sign_id}'. "
                f"load_real={self.load_real}, load_unreal={self.load_unreal}, "
                f"real_paths={paths.get('real', [])[:1]}, "
                f"unreal_paths={paths.get('unreal', [])[:1]}"
            )
        return tuple(feats)


class NGTSingleVideoProcessor(VideoProcessor):
    """Loads the real (Bushuis) feature for a sign from the pair manifest.
    Returns a single numpy array (N_clips, 768) — no pairing.
    Used with the baseline config (NGTSingleVideoProcessor + DSAligner).
    """

    def __init__(self, config):
        super().__init__(config)
        with open(config.pair_manifest) as f:
            self.manifest = json.load(f)

    def __call__(self, video_id, *args, **kwargs):
        path = self.manifest[video_id]['real']
        return np.load(path)


class NGTPairAligner(DSAligner):
    """Aligner for paired NGT data.

    Accepts a tuple of N video features and pads each independently to
    max_video_len, producing:
        vfeats:  (N, max_video_len, 768)
        vmasks:  (N, max_video_len)

    N = 3 for the baseline (load_unreal=False, piotrAnims views only) or
    N = 9 for the intervention (load_unreal=True, 3 real + 6 unreal).
    These are collated to (B, N, max_video_len, 768) and (B, N, max_video_len).
    NGTPairTask handles the reshape before the model forward pass.
    """

    def __call__(self, video_id, video_feature, text_feature, wps=0.7):
        # video_feature is a tuple of N numpy arrays, each (N_clips, 768)
        paired = [self._build_video_seq(f) for f in video_feature]
        vfeats = torch.stack([pf[0] for pf in paired])   # (N, T, 768)
        vmasks = torch.stack([pf[1] for pf in paired])   # (N, T)

        # Build a minimal dummy text (empty string → [CLS][SEP] tokens)
        text_feature_dict = {"cap": [text_feature], "start": [0], "end": [1]}
        caps, cmasks = self._build_text_seq(text_feature_dict, [0])

        return {
            "caps": caps,
            "cmasks": cmasks,
            "vfeats": vfeats,
            "vmasks": vmasks,
            "video_id": video_id,
        }


# ─── NGT end-to-end: raw frame processors ────────────────────────────────────
# Used with MMFusionE2E + NGTEndToEndTask.
# Instead of loading pre-extracted .npy features, these processors load raw
# video files (MKV) or PNG frame-sequence directories and apply the same
# preprocessing pipeline as extract_logos_features*.py.
#
# Requires a *raw* pair manifest (generated by make_ngt_raw_manifest.py) that
# maps sign_id → {real: [<mkv_path>, ...], unreal: [<png_dir>, ...]}.
#
# Config fields required:
#   pair_manifest: /path/to/ngt_raw_manifest.json
#   load_unreal:   true | false
#   max_video_len: 4   (set small — typical NGT sentence has 1–4 clips)

# Logos preprocessing constants — must match extract_logos_features*.py exactly
_LOGOS_CLIP_LEN       = 32
_LOGOS_FRAME_INTERVAL = 2
_LOGOS_CLIP_STRIDE    = 32
_LOGOS_RESIZE         = 300
_LOGOS_INPUT_SIZE     = 224
_LOGOS_MEAN = np.array([140.99762122, 129.92701646, 125.25081198], dtype=np.float32)
_LOGOS_STD  = np.array([62.07248248,  62.94645644,  61.42221137],  dtype=np.float32)


class NGTRawPairVideoProcessor(VideoProcessor):
    """Loads raw frames for each sign view and returns MViT-ready clip arrays.

    Accepts both:
      - MKV / MP4 video files   (piotrAnims real recordings)
      - PNG frame-sequence dirs  (Unreal Engine renders)

    Returns a tuple of K numpy arrays, each (N_clips, 3, 32, 224, 224) float32.
    """

    def __init__(self, config):
        super().__init__(config)
        with open(config.pair_manifest) as f:
            self.manifest = json.load(f)
        self.load_unreal = getattr(config, 'load_unreal', True)

    # ------------------------------------------------------------------

    @staticmethod
    def _load_frames(path):
        """Read all frames from a PNG directory or a video file (MKV/MP4)."""
        import cv2
        from pathlib import Path as _Path

        frames = []
        path = str(path)
        if os.path.isdir(path):
            for p in sorted(_Path(path).glob('*.png')):
                img = cv2.imread(str(p))
                if img is not None:
                    frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            cap = cv2.VideoCapture(path)
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
        return frames

    @staticmethod
    def _frames_to_clips(frames):
        """Preprocess frames and build sliding-window clips.

        Returns: (N_clips, 3, 32, 224, 224) float32
        """
        import cv2

        if not frames:
            return np.zeros(
                (1, 3, _LOGOS_CLIP_LEN, _LOGOS_INPUT_SIZE, _LOGOS_INPUT_SIZE),
                dtype=np.float32,
            )

        processed = []
        for frame in frames:
            h, w = frame.shape[:2]
            scale = _LOGOS_RESIZE / min(h, w)
            nh, nw = int(round(h * scale)), int(round(w * scale))
            frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
            pad_h = max(0, _LOGOS_RESIZE - nh)
            pad_w = max(0, _LOGOS_RESIZE - nw)
            frame = np.pad(
                frame,
                ((pad_h // 2, pad_h - pad_h // 2),
                 (pad_w // 2, pad_w - pad_w // 2),
                 (0, 0)),
                mode='constant',
            )
            y0 = (frame.shape[0] - _LOGOS_INPUT_SIZE) // 2
            x0 = (frame.shape[1] - _LOGOS_INPUT_SIZE) // 2
            frame = frame[y0:y0 + _LOGOS_INPUT_SIZE, x0:x0 + _LOGOS_INPUT_SIZE]
            frame = (frame.astype(np.float32) - _LOGOS_MEAN) / _LOGOS_STD
            processed.append(frame.transpose(2, 0, 1))   # (3, H, W)

        processed = np.stack(processed)   # (T, 3, H, W)
        T = len(processed)
        span = _LOGOS_CLIP_LEN * _LOGOS_FRAME_INTERVAL  # 64 raw frames per clip

        if T < span:
            starts = [0]
        else:
            starts = list(range(0, T - span + 1, _LOGOS_CLIP_STRIDE))
            if not starts:
                starts = [0]

        clips = []
        for s in starts:
            idx = [min(s + i * _LOGOS_FRAME_INTERVAL, T - 1)
                   for i in range(_LOGOS_CLIP_LEN)]
            clip = processed[idx].transpose(1, 0, 2, 3)   # (3, CLIP_LEN, H, W)
            clips.append(clip)

        return np.stack(clips)   # (N_clips, 3, 32, 224, 224)

    def __call__(self, sign_id):
        paths = self.manifest[sign_id]
        views = [self._frames_to_clips(self._load_frames(p))
                 for p in paths['real']]
        if self.load_unreal:
            views += [self._frames_to_clips(self._load_frames(p))
                      for p in paths.get('unreal', [])]
        return tuple(views)   # K views, each (N_clips, 3, 32, 224, 224)


class NGTRawPairAligner(DSAligner):
    """Pads raw-clip views to max_video_len and stacks into a single tensor.

    Produces:
        vfeats: (K, max_video_len, 3, 32, 224, 224)  float32
        vmasks: (K, max_video_len)                    bool

    Set max_video_len: 4 in the config — typical NGT sentences produce 1–4 clips.
    """

    def __call__(self, video_id, video_feature, text_feature, wps=0.7):
        T = self.max_video_len
        padded, masks = [], []
        for view in video_feature:   # each view: (N_clips, 3, 32, 224, 224)
            n = min(view.shape[0], T)
            pad = np.zeros(
                (T, 3, _LOGOS_CLIP_LEN, _LOGOS_INPUT_SIZE, _LOGOS_INPUT_SIZE),
                dtype=np.float32,
            )
            pad[:n] = view[:n]
            mask = np.zeros(T, dtype=bool)
            mask[:n] = True
            padded.append(pad)
            masks.append(mask)

        vfeats = torch.from_numpy(np.stack(padded))   # (K, T, 3, 32, 224, 224)
        vmasks = torch.from_numpy(np.stack(masks))    # (K, T)

        text_feature_dict = {"cap": [text_feature], "start": [0], "end": [1]}
        caps, cmasks = self._build_text_seq(text_feature_dict, [0])

        return {
            "caps": caps,
            "cmasks": cmasks,
            "vfeats": vfeats,
            "vmasks": vmasks,
            "video_id": video_id,
        }
