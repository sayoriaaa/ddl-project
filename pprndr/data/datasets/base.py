#  Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License")
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import abc

import paddle

from pprndr.cameras import Cameras

__all__ = ["BaseDataset"]


class _ParseCaller(type):
    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj._parse()
        return obj


class BaseDataset(paddle.io.Dataset, metaclass=_ParseCaller):
    @abc.abstractmethod
    def _parse(self) -> dict:
        """
        Parse dataset. This is automatically called when create a dataset as a post-initialization method.
        """

    @property
    @abc.abstractmethod
    def cameras(self) -> Cameras:
        """
        Cameras.
        """

    @property
    @abc.abstractmethod
    def split(self) -> str:
        """
        Which split ("train", "val", "trainval", etc.) the dataset belongs to.
        """

    @property
    def is_train_mode(self):
        return "train" in self.split
