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

from typing import Dict, Tuple, Union, List
import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from pprndr.apis import manager
from pprndr.cameras.rays import RaySamples
from pprndr.models.fields import BaseField
from pprndr.models.encoders import NeRFEncoder
from pprndr.models.layers import MLP

__all__ = ['DVGOGrid']


@manager.FIELDS.add_component
class DVGOGrid(BaseField):
    """
    Direct Voxel Grid Optimization. Reference: https://arxiv.org/abs/2111.11215

    Args:
        aabb: The aabb bounding box of the dataset.
        fea_encoder: The encoding method used for appearance encoding outputs.
        dir_encoder: The encoding method used for ray direction.
        density_encoder: The tensor encoding method used for scene density.
        color_encoder: The tensor encoding method used for scene color.
        color_head: Color network.
        appearance_dim: The number of dimensions for the appearance embedding.
        use_sh: Used spherical harmonics as the feature decoding function.
        sh_levels: The number of levels to use for spherical harmonics.
    """

    def __init__(self,
                 aabb: Union[paddle.Tensor, List],
            
                 alpha_init_coarse: float = 1e-6,
                 alpha_init_fine: float = 1e-2,
                 init_resolution_coarse: int = 100,
                 init_resolution_fine: int = 160,
                 k0_dim_fine: int = 12,

                 viewbase_pe: int = 4,          
                 rgbnet_hidden_num: int = 128,
                 rgbnet_number_layers: int = 2,
                 sh_levels: int = 2):
        super(DVGOGrid, self).__init__()

        self.aabb = paddle.to_tensor(aabb, dtype="float32").reshape([-1, 3])
      
        self.rgb_net = MLP(
            input_dim = 3 + 3*viewbase_pe*2 + k0_dim_fine,
            output_dim = 3,
            hidden_dim = rgbnet_hidden_num,
            num_layers = rgbnet_number_layers,
            activation = 'relu',
            output_activation = 'sigmoid',
        )

        self.fea_encoder = NeRFEncoder(
            min_freq = 0,
            max_freq = viewbase_pe - 1,
            num_freqs = viewbase_pe,
            include_identity = True
        )

        self.resolution_coarse = init_resolution_coarse
        self.resolution_fine = init_resolution_fine

        
        self.alpha_init_coarse = alpha_init_coarse
        self.alpha_init_fine = alpha_init_fine

        self.in_coarse_stage = True

        self.k0_dim_fine = k0_dim_fine
        self.k0_dim_coarse = 3 # rgb

        # build coarse stage grid
        self.feat_c = self.create_parameter(
            [init_resolution_coarse**3, self.k0_dim_coarse],
            attr=paddle.ParamAttr(
                initializer=nn.initializer.Constant(0.0),
                learning_rate=150 * (init_resolution_coarse**1.75)), # TODO: chage to pervoxel adam
            dtype="float32")
        self.densities_c = self.create_parameter(
            [init_resolution_coarse**3],
            attr=paddle.ParamAttr(
                initializer=nn.initializer.Constant(0.1),
                learning_rate=51.5 * (init_resolution_coarse**2.37)),
            dtype="float32")
        
        coarse_grid_ids = paddle.arange(init_resolution_coarse**3).reshape(
            [init_resolution_coarse] * 3)
        self.register_buffer("coarse_grid_ids", coarse_grid_ids, persistable=True)

        self.coarse_voxel_size = 
        
        # build fine stage grid
        self.feat_f = self.create_parameter(
            [init_resolution_fine**3, self.k0_dim_fine],
            attr=paddle.ParamAttr(
                initializer=nn.initializer.Constant(0.0),
                learning_rate=150 * (init_resolution_fine**1.75)), # TODO: chage to pervoxel adam
            dtype="float32")
        self.densities_f = self.create_parameter(
            [init_resolution_fine**3],
            attr=paddle.ParamAttr(
                initializer=nn.initializer.Constant(0.1),
                learning_rate=51.5 * (init_resolution_fine**2.37)),
            dtype="float32")     
        
        fine_grid_ids = paddle.arange(init_resolution_fine**3).reshape(
            [init_resolution_fine] * 3)
        self.register_buffer("fine_grid_ids", fine_grid_ids, persistable=True)


        
    
    
    def soft_plus_bias(self, alpha_init: float):
        return math.log((1-alpha_init)**(-1/self.voxel_size)-1) # Eq.(9)

    def get_density(self, ray_samples: Union[RaySamples, paddle.Tensor]
                    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        if isinstance(ray_samples, RaySamples):
            pos_inputs = ray_samples.frustums.positions
        else:
            pos_inputs = ray_samples
        positions = self.get_normalized_positions(pos_inputs)
        positions = positions * 2 - 1

        density_raw = self.density_encoder(positions)

        # dvgo post activation
        soft_plus = F.softplus()

        alpha_init = self.alpha_init_fine
        if self.in_coarse_stage:
            alpha_init = self.alpha_init_coarse

        density = soft_plus(density_raw + self.soft_plus_bias(alpha_init))

        return density, positions

    def density_L1(self):
        if isinstance(self.density_encoder, TensorCPEncoder):
            density_L1_loss = paddle.mean(paddle.abs(self.density_encoder.line_coef)) + \
                      paddle.mean(paddle.abs(self.color_encoder.line_coef))
        elif isinstance(self.density_encoder, TensorVMEncoder):
            density_L1_loss = paddle.mean(paddle.abs(self.density_encoder.line_coef)) + \
                      paddle.mean(paddle.abs(self.color_encoder.line_coef)) + \
                      paddle.mean(paddle.abs(self.density_encoder.plane_coef)) + \
                      paddle.mean(paddle.abs(self.color_encoder.plane_coef))
        elif isinstance(self.density_encoder, TriplaneEncoder):
            density_L1_loss = paddle.mean(paddle.abs(self.density_encoder.plane_coef)) + \
                      paddle.mean(paddle.abs(self.color_encoder.plane_coef))

        return density_L1_loss

    def get_outputs(self, ray_samples: RaySamples,
                    geo_features: paddle.Tensor) -> Dict[str, paddle.Tensor]:
        positions = self.get_normalized_positions(
            ray_samples.frustums.positions)
        d = ray_samples.frustums.directions
        positions = positions * 2 - 1
        rgb_features = self.color_encoder(positions)
        rgb_features = self.B(rgb_features)

        d_encoded = self.dir_encoder(d)
        rgb_features_encoded = self.fea_encoder(rgb_features)

        if self.use_sh:
            sh_mult = paddle.unsqueeze(self.sh(d), axis=-1)
            rgb_sh = rgb_features.reshape([sh_mult.shape[0], -1, 3])
            color = F.relu(paddle.sum(sh_mult * rgb_sh, axis=-2) + 0.5)
        else:
            color = self.color_head(
                paddle.concat(
                    [rgb_features, d, rgb_features_encoded, d_encoded],
                    axis=-1))  # type: ignore
        return dict(rgb=color)

    def get_normalized_positions(self, positions: paddle.Tensor):
        """Return normalized positions in range [0, 1] based on the aabb axis-aligned bounding box.

        Args:
            positions: the xyz positions
            aabb: the axis-aligned bounding box
        """
        aabb_lengths = self.aabb[1] - self.aabb[0]
        normalized_positions = (positions - self.aabb[0]) / aabb_lengths
        return normalized_positions

    def forward(self, ray_samples: RaySamples) -> Dict[str, paddle.Tensor]:
        density, _ = self.get_density(ray_samples)
        output = self.get_outputs(ray_samples, None)
        rgb = output["rgb"]

        return {"density": density, "rgb": rgb}