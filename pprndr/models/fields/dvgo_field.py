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
                 rgbnet_type: str = 'rgb_direct',
                 sh_levels: int = 2):
        super(DVGOGrid, self).__init__()

        self.aabb = paddle.to_tensor(aabb, dtype="float32").reshape([-1, 3]) # -> (2,3)

        self.rgbnet_type = rgbnet_type
        if rgbnet_type == 'rgb_direct':
            rgb_input_dim = 3 + 3*viewbase_pe*2 + k0_dim_fine
        if rgbnet_type == 'rgb_spec':
            rgb_input_dim = 3 + 3*viewbase_pe*2 + k0_dim_fine - 3
      
        self.rgb_net = MLP(
            input_dim = rgb_input_dim,
            output_dim = 3,
            hidden_dim = rgbnet_hidden_num,
            num_layers = rgbnet_number_layers,
            activation = 'relu',
            output_activation = 'sigmoid',
        )

        self.dir_encoder = NeRFEncoder(
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

        self.coarse_voxel_size = (paddle.prod(self.aabb[1] - self.aabb[0]).item() / self.resolution_coarse**3)**(1/3)    
        
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

        self.fine_voxel_size = (paddle.prod(self.aabb[1] - self.aabb[0]).item() / self.resolution_fine**3)**(1/3)    

        # determine stage
        self.stage = 'coarse'
        self.voxel_size = self.coarse_voxel_size
        self.resolution = self.resolution_coarse

    def go_go_fine_stage(self):
        self.stage = 'fine'
        self.voxel_size = self.fine_voxel_size
        self.resolution =self.resolution_fine
        
    
    
    def soft_plus_bias(self, alpha_init: float):
        return math.log((1-alpha_init)**(-1/self.voxel_size)-1) # Eq.(9)

    def _get_neighbors(self, positions: paddle.Tensor
                       ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        neighbor_offsets = paddle.to_tensor(
            [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1],
             [1, -1, 1], [1, 1, -1], [1, 1, 1]],
            dtype="float32") * self.voxel_size / 2.0  # [8, 3]
        
        direct_neighbors = positions.unsqueeze(-2) + neighbor_offsets.unsqueeze(0) # without considering boundary [N, 8, 3]
        neighbors = paddle.stack([
            paddle.clip(direct_neighbors[...,0],self.aabb[0,0].item(),self.aabb[1,0].item()),
            paddle.clip(direct_neighbors[...,1],self.aabb[0,1].item(),self.aabb[1,1].item()),
            paddle.clip(direct_neighbors[...,2],self.aabb[0,2].item(),self.aabb[1,2].item())],axis=-1)
        
        direct_neighbor_centers = (paddle.floor(neighbors / self.voxel_size + 1e-5) + 0.5) * self.voxel_size
        neighbor_centers = paddle.stack([
            paddle.clip(direct_neighbor_centers[...,0],self.aabb[0,0].item()+ self.voxel_size / 2, self.aabb[1,0].item() - self.voxel_size / 2),
            paddle.clip(direct_neighbor_centers[...,1],self.aabb[0,1].item()+ self.voxel_size / 2, self.aabb[1,1].item() - self.voxel_size / 2),
            paddle.clip(direct_neighbor_centers[...,2],self.aabb[0,2].item()+ self.voxel_size / 2, self.aabb[1,2].item() - self.voxel_size / 2)],axis=-1) # [N, 8, 3]
        
        neighbor_indices = (
            paddle.floor(neighbor_centers / self.voxel_size + 1e-5)
            + self.resolution / 2.0).astype("int32").clip(
                0, self.resolution - 1)  # [N, 8, 3]

        return neighbor_centers, neighbor_indices
    
    def _lookup(self,
                indices: paddle.Tensor) -> Tuple[paddle.Tensor, paddle.Tensor]:
        selected_ids = paddle.gather_nd(self.grid_ids, indices)  # [N, 8]
        empty_mask = selected_ids < 0.  # empty voxels have negative ids
        selected_ids = paddle.clip(selected_ids, min=0)

        if self.stage == 'coarse':
            neighbor_densities = paddle.gather_nd(self.densities_c,
                                              selected_ids[..., None])
            neighbor_densities[empty_mask] = 0.

            neighbor_feat = paddle.gather_nd(self.feat_c,
                                              selected_ids[..., None])
            neighbor_feat[empty_mask] = 0.

        if self.stage == 'fine':
            neighbor_densities = paddle.gather_nd(self.densities_f,
                                              selected_ids[..., None])
            neighbor_densities[empty_mask] = 0.

            neighbor_feat = paddle.gather_nd(self.feat_f,
                                              selected_ids[..., None])
            neighbor_feat[empty_mask] = 0.
        

        return neighbor_densities, neighbor_feat
    
    def _get_trilinear_interp_weights(self, interp_offset):
        """
        interp_offset: [N, num_intersections, 3], the offset (as a fraction of voxel_len)
            from the first (000) interpolation point.
        """
        interp_offset_x = interp_offset[..., 0]  # [N]
        interp_offset_y = interp_offset[..., 1]  # [N]
        interp_offset_z = interp_offset[..., 2]  # [N]
        weight_000 = (1 - interp_offset_x) * (1 - interp_offset_y) * (
            1 - interp_offset_z)
        weight_001 = (1 - interp_offset_x) * (
            1 - interp_offset_y) * interp_offset_z
        weight_010 = (1 - interp_offset_x) * interp_offset_y * (
            1 - interp_offset_z)
        weight_011 = (1 - interp_offset_x) * interp_offset_y * interp_offset_z
        weight_100 = interp_offset_x * (1 - interp_offset_y) * (
            1 - interp_offset_z)
        weight_101 = interp_offset_x * (1 - interp_offset_y) * interp_offset_z
        weight_110 = interp_offset_x * interp_offset_y * (1 - interp_offset_z)
        weight_111 = interp_offset_x * interp_offset_y * interp_offset_z

        weights = paddle.stack([
            weight_000, weight_001, weight_010, weight_011, weight_100,
            weight_101, weight_110, weight_111
        ],
                               axis=-1)  # [N, 8]

        return weights

    def _trilinear_interpolation(
            self, positions: paddle.Tensor, neighbor_centers: paddle.Tensor,
            neighbor_densities: paddle.Tensor, neighbor_feat: paddle.Tensor
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        interp_offset = (
            positions - neighbor_centers[..., 0, :]) / self.voxel_size  # [N, 3]
        interp_weights = self._get_trilinear_interp_weights(
            interp_offset)  # [N, 8]

        densities = paddle.sum(
            interp_weights * neighbor_densities, axis=-1,
            keepdim=True)  # [N, 1]
        feat = paddle.sum(
            interp_weights[..., None] * neighbor_feat,
            axis=-2)  # [N, k0_dim]

        return densities, feat
    
    def get_density(self, ray_samples: Union[RaySamples, paddle.Tensor]
                    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        if isinstance(ray_samples, RaySamples):
            positions = ray_samples.frustums.positions
        else:
            positions = ray_samples

        # dvgo post activation
        # Find the 8 neighbors of each positions
        neighbor_centers, neighbor_indices = self._get_neighbors(
            positions)  # [N, 8, 3], [N, 8, 3]

        # Look up neighbors' densities and SH coefficients
        neighbor_densities, neighbor_feat = self._lookup(
            neighbor_indices)  # [N, 8], [N, 8, k0_dim]

        # Tri-linear interpolation
        densities_raw, feat = self._trilinear_interpolation(
            positions, neighbor_centers, neighbor_densities,
            neighbor_feat)  # [N, 1], [N, k0_dim]

        # apply softplus
        soft_plus = F.softplus()

        alpha_init = self.alpha_init_fine
        if self.in_coarse_stage:
            alpha_init = self.alpha_init_coarse

        density = soft_plus(densities_raw + self.soft_plus_bias(alpha_init))

        return density, feat

    def get_outputs(self, ray_samples: RaySamples,
                    feat: paddle.Tensor) -> Dict[str, paddle.Tensor]:
        
        if self.stage == 'coarse':
            color = F.sigmoid(feat)
        if self.stage == 'fine':
            d = ray_samples.frustums.directions
            if self.rgbnet_type == 'rgb_direct':
                k0_view = feat 
            if self.rgbnet_type == 'rgb_spec':
                k0_view = feat[:, 3:]

            k0_diffuse = feat[:, :3]
            d_encoded = self.dir_encoder(d)
        
            rgb_feat = paddle.concat([k0_view, d_encoded], axis=-1)
            rgb_logit = self.rgb_net(rgb_feat)

            if self.rgbnet_type == 'rgb_direct':
                color = F.sigmoid(rgb_logit)
            if self.rgbnet_type == 'rgb_spec':
                color = F.sigmoid(rgb_logit + k0_diffuse)
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