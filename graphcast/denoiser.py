# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Support for wrapping a general Predictor to act as a Denoiser."""
# 한글 주석: 일반 예측기를 감싸 노이즈 제거기(Denoiser)로 작동하도록 지원하는 모듈입니다.

import dataclasses
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple # 한글 주석: 타입 힌트를 위한 모듈 임포트

import chex # JAX 및 NumPy를 위한 유틸리티 라이브러리
from graphcast import deep_typed_graph_net
from graphcast import denoisers_base as base # 한글 주석: Denoiser 기본 클래스 임포트
from graphcast import grid_mesh_connectivity
from graphcast import icosahedral_mesh # 정이십면체 메쉬 관련 모듈
from graphcast import model_utils # 모델 유틸리티 함수
from graphcast import sparse_transformer # 희소 트랜스포머 관련 모듈
from graphcast import transformer # 트랜스포머 관련 모듈
from graphcast import typed_graph
from graphcast import xarray_jax # Xarray와 JAX 연동 유틸리티
import haiku as hk # Haiku: JAX를 위한 신경망 라이브러리
import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse # 희소 행렬 라이브러리
import xarray


Kwargs = Mapping[str, Any] # 키워드 인자 타입 정의
NoiseLevelEncoder = Callable[[jnp.ndarray], jnp.ndarray] # 노이즈 레벨 인코더 함수 타입 정의


class FourierFeaturesMLP(hk.Module):
  """A simple MLP applied to Fourier features of values or their logarithms."""
  # 한글 주석: 값 또는 값의 로그에 대한 푸리에 특징에 적용되는 간단한 MLP입니다.

  def __init__(self,
               base_period: float,
               num_frequencies: int,
               output_sizes: Sequence[int],
               apply_log_first: bool = False,
               w_init: ... = None, # 가중치 초기화 (타입 힌트 생략)
               activation: ... = jax.nn.gelu, # 활성화 함수 (타입 힌트 생략)
               **mlp_kwargs
               ):
    """Initializes the module.
    # 한글 주석: 모듈을 초기화합니다.

    Args:
    # 한글 주석: 인수
      base_period:
        See model_utils.fourier_features. Note this would apply to log inputs if
        apply_log_first is used.
      # 한글 주석: base_period: model_utils.fourier_features를 참조하십시오. apply_log_first가 사용되면 로그 입력에 적용됩니다.
      num_frequencies:
        See model_utils.fourier_features.
      # 한글 주석: num_frequencies: model_utils.fourier_features를 참조하십시오.
      output_sizes:
        Layer sizes for the MLP.
      # 한글 주석: output_sizes: MLP의 계층 크기입니다.
      apply_log_first:
        Whether to take the log of the inputs before computing Fourier features.
      # 한글 주석: apply_log_first: 푸리에 특징을 계산하기 전에 입력의 로그를 취할지 여부입니다.
      w_init:
        Weights initializer for the MLP, default setting aims to produce
        approx unit-variance outputs given the input sin/cos features.
      # 한글 주석: w_init: MLP의 가중치 초기화 프로그램입니다. 기본 설정은 입력 sin/cos 특징이 주어졌을 때 대략 단위 분산 출력을 생성하는 것을 목표로 합니다.
      activation:
      # 한글 주석: activation: 활성화 함수입니다.
      **mlp_kwargs:
        Further settings for the MLP.
      # 한글 주석: **mlp_kwargs: MLP에 대한 추가 설정입니다.
    """
    super().__init__()
    self._base_period = base_period
    self._num_frequencies = num_frequencies
    self._apply_log_first = apply_log_first
    if w_init is None:
      # Scale of 2 is appropriate for input layer as sin/cos fourier features
      # have variance 0.5 for random inputs. Also reasonable to use for later
      # layers as relu activation cuts variance in half for inputs to later
      # layers and gelu something close enough too.
      # 한글 주석: sin/cos 푸리에 특징은 임의 입력에 대해 분산이 0.5이므로 입력 계층에 스케일 2가 적절합니다.
      # 또한 relu 활성화는 이후 계층에 대한 입력의 분산을 절반으로 줄이고 gelu도 충분히 가깝기 때문에 이후 계층에 사용하는 것도 합리적입니다.
      w_init = hk.initializers.VarianceScaling(
          2.0, mode="fan_in", distribution="uniform"
      )
    self._mlp = hk.nets.MLP(
        output_sizes=output_sizes,
        w_init=w_init,
        activation=activation,
        **mlp_kwargs)

  def __call__(self, values: jnp.ndarray) -> jnp.ndarray:
    # 한글 주석: MLP를 통해 푸리에 특징을 처리합니다.
    if self._apply_log_first: # 로그를 먼저 적용하는 경우
      values = jnp.log(values)

    # 푸리에 특징 계산
    features = model_utils.fourier_features(
        values, self._base_period, self._num_frequencies)

    return self._mlp(features) # MLP 통과


@chex.dataclass(frozen=True, eq=True)
class NoiseEncoderConfig:
  """Configures the noise level encoding.
  # 한글 주석: 노이즈 레벨 인코딩 설정을 구성합니다.

  Properties:
  # 한글 주석: 속성
    apply_log_first: Whether to take the log of the inputs before computing
      Fourier features.
    # 한글 주석: apply_log_first: 푸리에 특징을 계산하기 전에 입력의 로그를 취할지 여부입니다.
    base_period: The base period to use. This should be greater or equal to the
      range of the values, or to the period if the values have periodic
      semantics (e.g. 2pi if they represent angles). Frequencies used will be
      integer multiples of 1/base_period.
    # 한글 주석: base_period: 사용할 기본 주기입니다. 값의 범위보다 크거나 같아야 하며, 값이 주기적인 의미를 갖는 경우(예: 각도를 나타내는 경우 2pi) 주기와 같아야 합니다. 사용되는 주파수는 1/base_period의 정수배가 됩니다.
    num_frequencies: The number of frequencies to use, we will use integer
      multiples of 1/base_period from 1 up to num_frequencies inclusive. (We
      don't include a zero frequency as this would just give constant features
      which are redundant if a bias term is present).
    # 한글 주석: num_frequencies: 사용할 주파수의 수입니다. 1/base_period의 정수배를 1부터 num_frequencies까지(포함) 사용합니다. (0 주파수는 편향 항이 있는 경우 중복되는 상수 특징을 제공하므로 포함하지 않습니다.)
    output_sizes: Layer sizes for the MLP.
    # 한글 주석: output_sizes: MLP의 계층 크기입니다.
  """
  apply_log_first: bool = True
  base_period: float = 16.0
  num_frequencies: int = 32
  # 2-layer MLP applied to Fourier features
  # 한글 주석: 푸리에 특징에 적용되는 2계층 MLP
  output_sizes: tuple[int, int] = (32, 16)


@chex.dataclass(eq=True)
class SparseTransformerConfig:
  """Sparse Transformer config."""
  # 한글 주석: 희소 트랜스포머 설정입니다.
  # Neighbours to attend to.
  # 한글 주석: 어텐션할 이웃의 수 (k-hop).
  attention_k_hop: int
  # Primary width, the number of channels on the carrier path.
  # 한글 주석: 주 폭, 캐리어 경로의 채널 수입니다.
  d_model: int
  # Depth, or num transformer blocks. One 'layer' is attn + ffw.
  # 한글 주석: 깊이 또는 트랜스포머 블록 수입니다. 하나의 '계층'은 어텐션 + 피드포워드입니다.
  num_layers: int = 16
  # Number of heads for self-attention.
  # 한글 주석: 셀프 어텐션 헤드 수입니다.
  num_heads: int = 4
  # Attention type.
  # 한글 주석: 어텐션 유형입니다.
  attention_type: str = "splash_mha"
  # mask type if splash attention being used.
  # 한글 주석: splash 어텐션 사용 시 마스크 유형입니다.
  mask_type: str = "lazy"
  block_q: int = 1024
  block_kv: int = 512
  block_kv_compute: int = 256
  block_q_dkv: int = 512
  block_kv_dkv: int = 1024
  block_kv_dkv_compute: int = 1024
  # Init scale for final ffw layer (divided by depth)
  # 한글 주석: 최종 피드포워드 계층의 초기화 스케일 (깊이로 나눔)
  ffw_winit_final_mult: float = 0.0
  # Init scale for mha w (divided by depth).
  # 한글 주석: 멀티 헤드 어텐션 가중치의 초기화 스케일 (깊이로 나눔)
  attn_winit_final_mult: float = 0.0
  # Number of hidden units in the MLP blocks. Defaults to 4 * d_model.
  # 한글 주석: MLP 블록의 은닉 유닛 수입니다. 기본값은 4 * d_model입니다.
  ffw_hidden: int = 2048
  # Name for haiku module.
  # 한글 주석: Haiku 모듈의 이름입니다.
  name: Optional[str] = None


@chex.dataclass(eq=True)
class DenoiserArchitectureConfig:
  """Defines the GenCast architecture.
  # 한글 주석: GenCast 아키텍처를 정의합니다.

  Properties:
  # 한글 주석: 속성
    sparse_transformer_config: Config for the mesh transformer.
    # 한글 주석: sparse_transformer_config: 메쉬 트랜스포머 설정입니다.
    mesh_size: How many refinements to do on the multi-mesh.
    # 한글 주석: mesh_size: 다중 메쉬에 적용할 개선 횟수입니다.
    latent_size: How many latent features to include in the various MLPs.
    # 한글 주석: latent_size: 다양한 MLP에 포함할 잠재 특징의 수입니다.
    hidden_layers: How many hidden layers for each MLP.
    # 한글 주석: hidden_layers: 각 MLP의 은닉층 수입니다.
    radius_query_fraction_edge_length: Scalar that will be multiplied by the
      length of the longest edge of the finest mesh to define the radius of
      connectivity to use in the Grid2Mesh graph. Reasonable values are
      between 0.6 and 1. 0.6 reduces the number of grid points feeding into
      multiple mesh nodes and therefore reduces edge count and memory use, but
      1 gives better predictions.
    # 한글 주석: radius_query_fraction_edge_length: 가장 미세한 메쉬의 가장 긴 에지 길이에 곱해져 Grid2Mesh 그래프에서 사용할 연결 반경을 정의하는 스칼라 값입니다.
    # 합리적인 값은 0.6에서 1 사이입니다. 0.6은 여러 메쉬 노드로 들어가는 그리드 포인트 수를 줄여 에지 수와 메모리 사용량을 줄이지만, 1은 더 나은 예측을 제공합니다.
    norm_conditioning_features: List of feature names which will be used to
      condition the GNN via norm_conditioning, rather than as regular
      features. If this is provided, the GNN has to support the
      `global_norm_conditioning` argument. For now it only supports global
      norm conditioning (e.g. the same vector conditions all edges and nodes
      normalization), which means features passed here must not have "lat" or
      "lon" axes. In the future it may support node level norm conditioning
      too.
    # 한글 주석: norm_conditioning_features: 일반 특징 대신 norm_conditioning을 통해 GNN을 조건화하는 데 사용될 특징 이름 목록입니다.
    # 이 값이 제공되면 GNN은 `global_norm_conditioning` 인수를 지원해야 합니다. 현재는 전역 정규화 조건화만 지원합니다
    # (예: 동일한 벡터가 모든 에지 및 노드 정규화를 조건화함). 즉, 여기에 전달된 특징은 "lat" 또는 "lon" 축을 가져서는 안 됩니다.
    # 향후에는 노드 수준 정규화 조건화도 지원할 수 있습니다.
    grid2mesh_aggregate_normalization: Optional constant to normalize the output
      of aggregate_edges_for_nodes_fn in the mesh2grid GNN. This can be used to
        reduce the shock the model undergoes when switching resolution, which
        increases the number of edges connected to a node.
    # 한글 주석: grid2mesh_aggregate_normalization: mesh2grid GNN에서 aggregate_edges_for_nodes_fn의 출력을 정규화하는 선택적 상수입니다.
    # 해상도 전환 시 모델이 겪는 충격을 줄이는 데 사용할 수 있으며, 이는 노드에 연결된 에지 수를 증가시킵니다.
    node_output_size: Size of the output node representations for
        each node type. For node types not specified here, the latent node
        representation from the output of the processor will be returned.
    # 한글 주석: node_output_size: 각 노드 타입에 대한 출력 노드 표현의 크기입니다. 여기에 지정되지 않은 노드 타입의 경우, 프로세서 출력의 잠재 노드 표현이 반환됩니다.
  """

  sparse_transformer_config: SparseTransformerConfig
  mesh_size: int
  latent_size: int = 512
  hidden_layers: int = 1
  radius_query_fraction_edge_length: float = 0.6
  norm_conditioning_features: tuple[str, ...] = ("noise_level_encodings",)
  grid2mesh_aggregate_normalization: Optional[float] = None
  node_output_size: Optional[int] = None


class Denoiser(base.Denoiser):
  """Wraps a general deterministic Predictor to act as a Denoiser.
  # 한글 주석: 일반적인 결정론적 예측기를 감싸 노이즈 제거기로 작동하도록 하는 래퍼 클래스입니다.

  This passes an encoding of the noise level as an additional input to the
  Predictor as an additional input 'noise_level_encodings' with shape
  ('batch', 'noise_level_encoding_channels'). It passes the noisy_targets as
  additional forcings (since they are also per-target-timestep data that the
  predictor needs to condition on) with the same names as the original target
  variables.
  # 한글 주석: 이 클래스는 노이즈 레벨 인코딩을 ('batch', 'noise_level_encoding_channels') 형태의 추가 입력 'noise_level_encodings'로 예측기에 전달합니다.
  # 또한 노이즈가 낀 목표(noisy_targets)를 원본 목표 변수와 동일한 이름으로 추가적인 강제 변수(forcings)로 전달합니다
  # (이는 예측기가 조건화해야 하는 목표 시간 단계별 데이터이기도 하기 때문입니다).
  """

  def __init__(
      self,
      noise_encoder_config: Optional[NoiseEncoderConfig], # 노이즈 인코더 설정
      denoiser_architecture_config: DenoiserArchitectureConfig, # 노이즈 제거기 아키텍처 설정
  ):
    # 내부 예측기로 _DenoiserArchitecture 사용
    self._predictor = _DenoiserArchitecture(
        denoiser_architecture_config=denoiser_architecture_config,
    )
    # Use default values if not specified.
    # 한글 주석: 지정되지 않은 경우 기본값을 사용합니다.
    if noise_encoder_config is None:
      noise_encoder_config = NoiseEncoderConfig() # 기본 노이즈 인코더 설정 사용
    self._noise_level_encoder = FourierFeaturesMLP(**noise_encoder_config) # 푸리에 특징 MLP를 노이즈 레벨 인코더로 사용

  def __call__(
      self,
      inputs: xarray.Dataset, # 입력 데이터셋
      noisy_targets: xarray.Dataset, # 노이즈가 낀 목표 데이터셋
      noise_levels: xarray.DataArray, # 노이즈 레벨
      forcings: Optional[xarray.Dataset] = None, # 추가적인 강제 변수 (선택 사항)
      **kwargs) -> xarray.Dataset:
    # 한글 주석: 노이즈 제거기를 호출하여 예측을 수행합니다.
    if forcings is None: forcings = xarray.Dataset() # 강제 변수가 없으면 빈 데이터셋으로 초기화
    forcings = forcings.assign(noisy_targets) # 강제 변수에 노이즈 낀 목표를 할당

    if noise_levels.dims != ("batch",): # 노이즈 레벨의 차원 확인
      raise ValueError("noise_levels는 ('batch',) 형태여야 합니다.")
    # 노이즈 레벨을 인코딩합니다.
    noise_level_encodings = self._noise_level_encoder(
        xarray_jax.unwrap_data(noise_levels) # xarray에서 JAX 배열로 변환
    )
    # 인코딩된 노이즈 레벨을 xarray 변수로 변환합니다.
    noise_level_encodings = xarray_jax.Variable(
        ("batch", "noise_level_encoding_channels"), noise_level_encodings
    )
    inputs = inputs.assign(noise_level_encodings=noise_level_encodings) # 입력에 노이즈 레벨 인코딩 추가

    # 내부 예측기 호출
    return self._predictor(
        inputs=inputs,
        targets_template=noisy_targets, # 목표 템플릿으로 노이즈 낀 목표 사용
        forcings=forcings,
        **kwargs)


class _DenoiserArchitecture:
  """GenCast Predictor.
  # 한글 주석: GenCast 예측기 아키텍처입니다. (실제로는 Denoiser의 내부 아키텍처)

  The model works on graphs that take into account:
  # 한글 주석: 이 모델은 다음을 고려하는 그래프에서 작동합니다:
  * Mesh nodes: nodes for the vertices of the mesh.
  # 한글 주석: * 메쉬 노드: 메쉬 정점의 노드입니다.
  * Grid nodes: nodes for the points of the grid.
  # 한글 주석: * 그리드 노드: 그리드 지점의 노드입니다.
  * Nodes: When referring to just "nodes", this means the joint set of
    both mesh nodes, concatenated with grid nodes.
  # 한글 주석: * 노드: 단순히 "노드"라고 할 때는 메쉬 노드와 그리드 노드를 연결한 전체 집합을 의미합니다.

  The model works with 3 graphs:
  # 한글 주석: 이 모델은 3개의 그래프로 작동합니다:
  * Grid2Mesh graph: Graph that contains all nodes. This graph is strictly
    bipartite with edges going from grid nodes to mesh nodes using a
    fixed radius query. The grid2mesh_gnn will operate in this graph. The output
    of this stage will be a latent representation for the mesh nodes, and a
    latent representation for the grid nodes.
  # 한글 주석: * Grid2Mesh 그래프: 모든 노드를 포함하는 그래프입니다. 이 그래프는 고정 반경 쿼리를 사용하여 그리드 노드에서 메쉬 노드로 향하는 에지를 가진 엄격한 이분 그래프입니다. grid2mesh_gnn이 이 그래프에서 작동합니다. 이 단계의 출력은 메쉬 노드에 대한 잠재 표현과 그리드 노드에 대한 잠재 표현이 됩니다.
  * Mesh graph: Graph that contains mesh nodes only. The mesh_gnn will
    operate in this graph. It will update the latent state of the mesh nodes
    only.
  # 한글 주석: * 메쉬 그래프: 메쉬 노드만 포함하는 그래프입니다. mesh_gnn이 이 그래프에서 작동하며 메쉬 노드의 잠재 상태만 업데이트합니다.
  * Mesh2Grid graph: Graph that contains all nodes. This graph is strictly
    bipartite with edges going from mesh nodes to grid nodes such that each grid
    node is connected to 3 nodes of the mesh triangular face that contains
    the grid points. The mesh2grid_gnn will operate in this graph. It will
    process the updated latent state of the mesh nodes, and the latent state
    of the grid nodes, to produce the final output for the grid nodes.
  # 한글 주석: * Mesh2Grid 그래프: 모든 노드를 포함하는 그래프입니다. 이 그래프는 각 그리드 노드가 그리드 지점을 포함하는 메쉬 삼각형 면의 3개 노드에 연결되도록 메쉬 노드에서 그리드 노드로 향하는 에지를 가진 엄격한 이분 그래프입니다. mesh2grid_gnn이 이 그래프에서 작동합니다. 메쉬 노드의 업데이트된 잠재 상태와 그리드 노드의 잠재 상태를 처리하여 그리드 노드에 대한 최종 출력을 생성합니다.

  The model is built on top of `TypedGraph`s so the different types of nodes and
  edges can be stored and treated separately.
  # 한글 주석: 이 모델은 `TypedGraph` 위에 구축되어 다양한 유형의 노드와 에지를 별도로 저장하고 처리할 수 있습니다.
  """

  def __init__(
      self,
      denoiser_architecture_config: DenoiserArchitectureConfig, # 노이즈 제거기 아키텍처 설정
  ):
    """Initializes the predictor."""
    # 한글 주석: 예측기를 초기화합니다.
    self._spatial_features_kwargs = dict( # 공간 특징 생성 시 사용할 키워드 인수들
        add_node_positions=False, # 노드 위치 추가 안 함
        add_node_latitude=True, # 노드 위도 추가
        add_node_longitude=True, # 노드 경도 추가
        add_relative_positions=True, # 상대 위치 추가
        relative_longitude_local_coordinates=True, # 상대 경도 로컬 좌표 사용
        relative_latitude_local_coordinates=True, # 상대 위도 로컬 좌표 사용
    )

    # Construct the mesh.
    # 한글 주석: 메쉬를 구성합니다.
    mesh = icosahedral_mesh.get_last_triangular_mesh_for_sphere( # 구에 대한 마지막 삼각형 메쉬 가져오기
        splits=denoiser_architecture_config.mesh_size # 메쉬 개선 횟수
    )
    # Permute the mesh to a banded structure so we can run sparse attention
    # operations.
    # 한글 주석: 희소 어텐션 연산을 실행할 수 있도록 메쉬를 띠 구조로 순열합니다.
    self._mesh = _permute_mesh_to_banded(mesh=mesh)

    # Encoder, which moves data from the grid to the mesh with a single message
    # passing step.
    # 한글 주석: 인코더, 단일 메시지 전달 단계를 통해 그리드에서 메쉬로 데이터를 이동시킵니다.
    self._grid2mesh_gnn = (
        deep_typed_graph_net.DeepTypedGraphNet( # 심층 타입 그래프 신경망 사용
            activation="swish", # 활성화 함수
            aggregate_normalization=( # 집계 정규화
                denoiser_architecture_config.grid2mesh_aggregate_normalization
            ),
            edge_latent_size=dict( # 에지 잠재 크기
                grid2mesh=denoiser_architecture_config.latent_size
            ),
            embed_edges=True, # 에지 임베딩 사용
            embed_nodes=True, # 노드 임베딩 사용
            f32_aggregation=True, # float32 집계 사용
            include_sent_messages_in_node_update=False, # 노드 업데이트 시 보낸 메시지 미포함
            mlp_hidden_size=denoiser_architecture_config.latent_size, # MLP 은닉 크기
            mlp_num_hidden_layers=denoiser_architecture_config.hidden_layers, # MLP 은닉층 수
            name="grid2mesh_gnn", # 모듈 이름
            node_latent_size=dict( # 노드 잠재 크기
                grid_nodes=denoiser_architecture_config.latent_size,
                mesh_nodes=denoiser_architecture_config.latent_size
            ),
            node_output_size=None, # 노드 출력 크기 (지정 안 함)
            num_message_passing_steps=1, # 메시지 전달 단계 수
            use_layer_norm=True, # 레이어 정규화 사용
            use_norm_conditioning=True, # 정규화 조건화 사용
        )
    )

    # Processor - performs multiple rounds of message passing on the mesh.
    # 한글 주석: 프로세서 - 메쉬에서 여러 라운드의 메시지 전달을 수행합니다.
    self._mesh_gnn = transformer.MeshTransformer( # 메쉬 트랜스포머 사용
        name="mesh_transformer", # 모듈 이름
        transformer_ctor=sparse_transformer.Transformer, # 희소 트랜스포머 생성자
        transformer_kwargs=dataclasses.asdict( # 트랜스포머 키워드 인수
            denoiser_architecture_config.sparse_transformer_config
        ),
    )

    # Decoder, which moves data from the mesh back into the grid with a single
    # message passing step.
    # 한글 주석: 디코더, 단일 메시지 전달 단계를 통해 메쉬에서 그리드로 데이터를 다시 이동시킵니다.
    self._mesh2grid_gnn = (
        deep_typed_graph_net.DeepTypedGraphNet(
            activation="swish",
            edge_latent_size=dict(
                mesh2grid=denoiser_architecture_config.latent_size
            ),
            embed_nodes=False, # 노드 임베딩 미사용 (이미 잠재 공간에 있음)
            f32_aggregation=False, # float32 집계 미사용
            include_sent_messages_in_node_update=False,
            mlp_hidden_size=denoiser_architecture_config.latent_size,
            mlp_num_hidden_layers=denoiser_architecture_config.hidden_layers,
            name="mesh2grid_gnn",
            node_latent_size=dict(
                grid_nodes=denoiser_architecture_config.latent_size,
                mesh_nodes=denoiser_architecture_config.latent_size,
            ),
            node_output_size={ # 그리드 노드의 출력 크기 지정
                "grid_nodes": denoiser_architecture_config.node_output_size
            },
            num_message_passing_steps=1,
            use_layer_norm=True,
            use_norm_conditioning=True,
        )
    )

    self._norm_conditioning_features = ( # 정규화 조건화에 사용할 특징 이름들
        denoiser_architecture_config.norm_conditioning_features
    )
    # Obtain the query radius in absolute units for the unit-sphere for the
    # grid2mesh model, by rescaling the `radius_query_fraction_edge_length`.
    # 한글 주석: grid2mesh 모델에 대한 단위 구의 절대 단위 쿼리 반경을 `radius_query_fraction_edge_length`를 재조정하여 얻습니다.
    self._query_radius = (
        _get_max_edge_distance(self._mesh) # 메쉬의 최대 에지 거리 계산
        * denoiser_architecture_config.radius_query_fraction_edge_length # 비율 곱하기
    )

    # Other initialization is delayed until the first call (`_maybe_init`)
    # when we get some sample data so we know the lat/lon values.
    # 한글 주석: 다른 초기화는 lat/lon 값을 알 수 있는 샘플 데이터를 얻는 첫 번째 호출(`_maybe_init`)까지 지연됩니다.
    self._initialized = False # 초기화 플래그

    # A "_init_mesh_properties":
    # This one could be initialized at init but we delay it for consistency too.
    # 한글 주석: "_init_mesh_properties": 초기화 시 수행될 수 있지만 일관성을 위해 지연됩니다.
    self._num_mesh_nodes = None  # 메쉬 노드 수
    self._mesh_nodes_lat = None  # 메쉬 노드 위도 [num_mesh_nodes]
    self._mesh_nodes_lon = None  # 메쉬 노드 경도 [num_mesh_nodes]

    # A "_init_grid_properties":
    # 한글 주석: "_init_grid_properties":
    self._grid_lat = None  # 그리드 위도 [num_lat_points]
    self._grid_lon = None  # 그리드 경도 [num_lon_points]
    self._num_grid_nodes = None  # 그리드 노드 수 (num_lat_points * num_lon_points)
    self._grid_nodes_lat = None  # 그리드 노드 위도 [num_grid_nodes]
    self._grid_nodes_lon = None  # 그리드 노드 경도 [num_grid_nodes]

    # A "_init_{grid2mesh,processor,mesh2grid}_graph"
    # 한글 주석: "_init_{grid2mesh,processor,mesh2grid}_graph":
    self._grid2mesh_graph_structure = None # Grid2Mesh 그래프 구조
    self._mesh_graph_structure = None # 메쉬 그래프 구조
    self._mesh2grid_graph_structure = None # Mesh2Grid 그래프 구조

  def __call__(self,
               inputs: xarray.Dataset,
               targets_template: xarray.Dataset,
               forcings: xarray.Dataset,
               ) -> xarray.Dataset:
    # 한글 주석: 예측기 호출 (순전파)
    self._maybe_init(inputs) # 필요시 초기화 수행

    # Convert all input data into flat vectors for each of the grid nodes.
    # xarray (batch, time, lat, lon, level, multiple vars, forcings)
    # -> [num_grid_nodes, batch, num_channels]
    # 한글 주석: 모든 입력 데이터를 각 그리드 노드에 대한 평탄한 벡터로 변환합니다.
    grid_node_features, global_norm_conditioning = (
        self._inputs_to_grid_node_features_and_norm_conditioning( # 입력 및 강제 변수에서 그리드 노드 특징과 전역 정규화 조건 추출
            inputs, forcings
        )
    )

    # [num_mesh_nodes, batch, latent_size], [num_grid_nodes, batch, latent_size]
    # 한글 주석: Grid2Mesh GNN 실행하여 메쉬 및 그리드 노드의 잠재 표현 추출
    (latent_mesh_nodes, latent_grid_nodes) = self._run_grid2mesh_gnn(
        grid_node_features, global_norm_conditioning
    )

    # Run message passing in the multimesh.
    # [num_mesh_nodes, batch, latent_size]
    # 한글 주석: 다중 메쉬에서 메시지 전달 실행 (메쉬 GNN)
    updated_latent_mesh_nodes = self._run_mesh_gnn(
        latent_mesh_nodes, global_norm_conditioning
    )

    # Transfer data from the mesh to the grid.
    # [num_grid_nodes, batch, output_size]
    # 한글 주석: 메쉬에서 그리드로 데이터 전송 (Mesh2Grid GNN)
    output_grid_nodes = self._run_mesh2grid_gnn(
        updated_latent_mesh_nodes, latent_grid_nodes, global_norm_conditioning
    )

    # Convert output flat vectors for the grid nodes to the format of the
    # output. [num_grid_nodes, batch, output_size] -> xarray (batch, one time
    # step, lat, lon, level, multiple vars)
    # 한글 주석: 그리드 노드에 대한 출력 평탄 벡터를 출력 형식으로 변환합니다.
    return self._grid_node_outputs_to_prediction(
        output_grid_nodes, targets_template
    )

  def _maybe_init(self, sample_inputs: xarray.Dataset):
    """Inits everything that has a dependency on the input coordinates."""
    # 한글 주석: 입력 좌표에 의존하는 모든 것을 초기화합니다 (필요한 경우).
    if not self._initialized: # 아직 초기화되지 않았다면
      self._init_mesh_properties() # 메쉬 속성 초기화
      self._init_grid_properties( # 그리드 속성 초기화
          grid_lat=sample_inputs.lat, grid_lon=sample_inputs.lon)
      self._grid2mesh_graph_structure = self._init_grid2mesh_graph() # Grid2Mesh 그래프 구조 초기화
      self._mesh_graph_structure = self._init_mesh_graph() # 메쉬 그래프 구조 초기화
      self._mesh2grid_graph_structure = self._init_mesh2grid_graph() # Mesh2Grid 그래프 구조 초기화

      self._initialized = True # 초기화 완료 플래그 설정

  def _init_mesh_properties(self):
    """Inits static properties that have to do with mesh nodes."""
    # 한글 주석: 메쉬 노드와 관련된 정적 속성을 초기화합니다.
    self._num_mesh_nodes = self._mesh.vertices.shape[0] # 메쉬 노드 수
    mesh_phi, mesh_theta = model_utils.cartesian_to_spherical( # 데카르트 좌표를 구면 좌표로 변환
        self._mesh.vertices[:, 0],
        self._mesh.vertices[:, 1],
        self._mesh.vertices[:, 2])
    (
        mesh_nodes_lat, # 메쉬 노드 위도
        mesh_nodes_lon, # 메쉬 노드 경도
    ) = model_utils.spherical_to_lat_lon( # 구면 좌표를 위도/경도로 변환
        phi=mesh_phi, theta=mesh_theta)
    # Convert to f32 to ensure the lat/lon features aren't in f64.
    # 한글 주석: lat/lon 특징이 f64가 아닌 f32인지 확인하기 위해 변환합니다.
    self._mesh_nodes_lat = mesh_nodes_lat.astype(np.float32)
    self._mesh_nodes_lon = mesh_nodes_lon.astype(np.float32)

  def _init_grid_properties(self, grid_lat: np.ndarray, grid_lon: np.ndarray):
    """Inits static properties that have to do with grid nodes."""
    # 한글 주석: 그리드 노드와 관련된 정적 속성을 초기화합니다.
    self._grid_lat = grid_lat.astype(np.float32) # 그리드 위도
    self._grid_lon = grid_lon.astype(np.float32) # 그리드 경도
    # Initialized the counters.
    # 한글 주석: 카운터를 초기화합니다.
    self._num_grid_nodes = grid_lat.shape[0] * grid_lon.shape[0] # 그리드 노드 수

    # Initialize lat and lon for the grid.
    # 한글 주석: 그리드의 위도와 경도를 초기화합니다.
    grid_nodes_lon, grid_nodes_lat = np.meshgrid(grid_lon, grid_lat) # 경도, 위도로부터 그리드 생성
    self._grid_nodes_lon = grid_nodes_lon.reshape([-1]).astype(np.float32) # 1D 배열로 변환
    self._grid_nodes_lat = grid_nodes_lat.reshape([-1]).astype(np.float32) # 1D 배열로 변환

  def _init_grid2mesh_graph(self) -> typed_graph.TypedGraph:
    """Build Grid2Mesh graph."""
    # 한글 주석: Grid2Mesh 그래프를 구축합니다.

    # Create some edges according to distance between mesh and grid nodes.
    # 한글 주석: 메쉬 노드와 그리드 노드 간의 거리에 따라 에지를 생성합니다.
    assert self._grid_lat is not None and self._grid_lon is not None
    (grid_indices, mesh_indices) = grid_mesh_connectivity.radius_query_indices( # 반경 쿼리를 사용하여 인덱스 찾기
        grid_latitude=self._grid_lat,
        grid_longitude=self._grid_lon,
        mesh=self._mesh,
        radius=self._query_radius) # 미리 계산된 쿼리 반경 사용

    # Edges sending info from grid to mesh.
    # 한글 주석: 그리드에서 메쉬로 정보를 보내는 에지입니다.
    senders = grid_indices # 송신 노드 (그리드)
    receivers = mesh_indices # 수신 노드 (메쉬)

    # Precompute structural node and edge features according to config options.
    # Structural features are those that depend on the fixed values of the
    # latitude and longitudes of the nodes.
    # 한글 주석: 설정 옵션에 따라 구조적 노드 및 에지 특징을 미리 계산합니다.
    # 구조적 특징은 노드의 위도 및 경도의 고정된 값에 따라 달라집니다.
    (senders_node_features, receivers_node_features,
     edge_features) = model_utils.get_bipartite_graph_spatial_features( # 이분 그래프 공간 특징 계산
         senders_node_lat=self._grid_nodes_lat,
         senders_node_lon=self._grid_nodes_lon,
         receivers_node_lat=self._mesh_nodes_lat,
         receivers_node_lon=self._mesh_nodes_lon,
         senders=senders,
         receivers=receivers,
         edge_normalization_factor=None, # 에지 정규화 인자 (여기서는 사용 안 함)
         **self._spatial_features_kwargs, # 공간 특징 생성 관련 인수들
     )

    n_grid_node = np.array([self._num_grid_nodes]) # 그리드 노드 수
    n_mesh_node = np.array([self._num_mesh_nodes]) # 메쉬 노드 수
    n_edge = np.array([mesh_indices.shape[0]]) # 에지 수
    # 그리드 노드 집합 생성
    grid_node_set = typed_graph.NodeSet(
        n_node=n_grid_node, features=senders_node_features)
    # 메쉬 노드 집합 생성
    mesh_node_set = typed_graph.NodeSet(
        n_node=n_mesh_node, features=receivers_node_features)
    # 에지 집합 생성
    edge_set = typed_graph.EdgeSet(
        n_edge=n_edge,
        indices=typed_graph.EdgesIndices(senders=senders, receivers=receivers),
        features=edge_features)
    nodes = {"grid_nodes": grid_node_set, "mesh_nodes": mesh_node_set} # 노드 딕셔너리
    edges = { # 에지 딕셔너리
        typed_graph.EdgeSetKey("grid2mesh", ("grid_nodes", "mesh_nodes")): # 에지 타입 키
            edge_set
    }
    # TypedGraph 객체 생성
    grid2mesh_graph = typed_graph.TypedGraph(
        context=typed_graph.Context(n_graph=np.array([1]), features=()), # 컨텍스트 (여기서는 비어 있음)
        nodes=nodes,
        edges=edges)
    return grid2mesh_graph

  def _init_mesh_graph(self) -> typed_graph.TypedGraph:
    """Build Mesh graph."""
    # 한글 주석: 메쉬 그래프를 구축합니다.
    # Work simply on the mesh edges.
    # N.B.To make sure ordering is preserved, any changes to faces_to_edges here
    # should be reflected in the other 2 calls to faces_to_edges in this file.
    # 한글 주석: 메쉬 에지에 대해 간단히 작업합니다.
    # 참고: 순서 보존을 위해 여기서 faces_to_edges에 대한 변경 사항은 이 파일의 다른 2개 faces_to_edges 호출에도 반영되어야 합니다.
    senders, receivers = icosahedral_mesh.faces_to_edges(self._mesh.faces) # 메쉬 면에서 에지 추출

    # Precompute structural node and edge features according to config options.
    # Structural features are those that depend on the fixed values of the
    # latitude and longitudes of the nodes.
    # 한글 주석: 설정 옵션에 따라 구조적 노드 및 에지 특징을 미리 계산합니다.
    # 구조적 특징은 노드의 위도 및 경도의 고정된 값에 따라 달라집니다.
    assert self._mesh_nodes_lat is not None and self._mesh_nodes_lon is not None
    node_features, edge_features = model_utils.get_graph_spatial_features( # 그래프 공간 특징 계산
        node_lat=self._mesh_nodes_lat,
        node_lon=self._mesh_nodes_lon,
        senders=senders,
        receivers=receivers,
        **self._spatial_features_kwargs,
    )

    n_mesh_node = np.array([self._num_mesh_nodes]) # 메쉬 노드 수
    n_edge = np.array([senders.shape[0]]) # 에지 수
    assert n_mesh_node == len(node_features) # 노드 수와 특징 길이 일치 확인
    # 메쉬 노드 집합 생성
    mesh_node_set = typed_graph.NodeSet(
        n_node=n_mesh_node, features=node_features)
    # 에지 집합 생성
    edge_set = typed_graph.EdgeSet(
        n_edge=n_edge,
        indices=typed_graph.EdgesIndices(senders=senders, receivers=receivers),
        features=edge_features)
    nodes = {"mesh_nodes": mesh_node_set} # 노드 딕셔너리
    edges = { # 에지 딕셔너리
        typed_graph.EdgeSetKey("mesh", ("mesh_nodes", "mesh_nodes")): edge_set # "mesh" 타입 에지
    }
    # TypedGraph 객체 생성
    mesh_graph = typed_graph.TypedGraph(
        context=typed_graph.Context(n_graph=np.array([1]), features=()),
        nodes=nodes,
        edges=edges)

    return mesh_graph

  def _init_mesh2grid_graph(self) -> typed_graph.TypedGraph:
    """Build Mesh2Grid graph."""
    # 한글 주석: Mesh2Grid 그래프를 구축합니다.

    # Create some edges according to how the grid nodes are contained by
    # mesh triangles.
    # 한글 주석: 그리드 노드가 메쉬 삼각형에 포함되는 방식에 따라 에지를 생성합니다.
    (grid_indices,
     mesh_indices) = grid_mesh_connectivity.in_mesh_triangle_indices( # 메쉬 삼각형 내 그리드 노드 인덱스 찾기
         grid_latitude=self._grid_lat,
         grid_longitude=self._grid_lon,
         mesh=self._mesh)

    # Edges sending info from mesh to grid.
    # 한글 주석: 메쉬에서 그리드로 정보를 보내는 에지입니다.
    senders = mesh_indices # 송신 노드 (메쉬)
    receivers = grid_indices # 수신 노드 (그리드)

    # Precompute structural node and edge features according to config options.
    # 한글 주석: 설정 옵션에 따라 구조적 노드 및 에지 특징을 미리 계산합니다.
    assert self._mesh_nodes_lat is not None and self._mesh_nodes_lon is not None
    (senders_node_features, receivers_node_features,
     edge_features) = model_utils.get_bipartite_graph_spatial_features(
         senders_node_lat=self._mesh_nodes_lat,
         senders_node_lon=self._mesh_nodes_lon,
         receivers_node_lat=self._grid_nodes_lat,
         receivers_node_lon=self._grid_nodes_lon,
         senders=senders,
         receivers=receivers,
         edge_normalization_factor=None,
         **self._spatial_features_kwargs,
     )

    n_grid_node = np.array([self._num_grid_nodes])
    n_mesh_node = np.array([self._num_mesh_nodes])
    n_edge = np.array([senders.shape[0]])
    grid_node_set = typed_graph.NodeSet(
        n_node=n_grid_node, features=receivers_node_features)
    mesh_node_set = typed_graph.NodeSet(
        n_node=n_mesh_node, features=senders_node_features)
    edge_set = typed_graph.EdgeSet(
        n_edge=n_edge,
        indices=typed_graph.EdgesIndices(senders=senders, receivers=receivers),
        features=edge_features)
    nodes = {"grid_nodes": grid_node_set, "mesh_nodes": mesh_node_set}
    edges = {
        typed_graph.EdgeSetKey("mesh2grid", ("mesh_nodes", "grid_nodes")): # "mesh2grid" 타입 에지
            edge_set
    }
    mesh2grid_graph = typed_graph.TypedGraph(
        context=typed_graph.Context(n_graph=np.array([1]), features=()),
        nodes=nodes,
        edges=edges)
    return mesh2grid_graph

  def _run_grid2mesh_gnn(self, grid_node_features: chex.Array,
                         global_norm_conditioning: Optional[chex.Array] = None,
                         ) -> tuple[chex.Array, chex.Array]:
    """Runs the grid2mesh_gnn, extracting latent mesh and grid nodes."""
    # 한글 주석: grid2mesh_gnn을 실행하여 잠재 메쉬 및 그리드 노드를 추출합니다.

    # Concatenate node structural features with input features.
    # 한글 주석: 노드 구조적 특징과 입력 특징을 연결합니다.
    batch_size = grid_node_features.shape[1] # 배치 크기

    grid2mesh_graph = self._grid2mesh_graph_structure # 미리 초기화된 그래프 구조 사용
    assert grid2mesh_graph is not None
    grid_nodes = grid2mesh_graph.nodes["grid_nodes"]
    mesh_nodes = grid2mesh_graph.nodes["mesh_nodes"]
    # 그리드 노드 특징 업데이트: 입력 특징 + 구조적 특징
    new_grid_nodes = grid_nodes._replace(
        features=jnp.concatenate([
            grid_node_features, # 동적 입력 특징
            _add_batch_second_axis( # 구조적 특징 (배치 차원 추가)
                grid_nodes.features.astype(grid_node_features.dtype),
                batch_size)
        ],
                                 axis=-1))

    # To make sure capacity of the embedded is identical for the grid nodes and
    # the mesh nodes, we also append some dummy zero input features for the
    # mesh nodes.
    # 한글 주석: 그리드 노드와 메쉬 노드의 임베딩 용량이 동일하도록 메쉬 노드에 더미 0 입력 특징을 추가합니다.
    dummy_mesh_node_features = jnp.zeros( # 메쉬 노드에 대한 더미 특징 (입력 특징과 동일한 형태)
        (self._num_mesh_nodes,) + grid_node_features.shape[1:],
        dtype=grid_node_features.dtype)
    # 메쉬 노드 특징 업데이트: 더미 특징 + 구조적 특징
    new_mesh_nodes = mesh_nodes._replace(
        features=jnp.concatenate([
            dummy_mesh_node_features,
            _add_batch_second_axis(
                mesh_nodes.features.astype(dummy_mesh_node_features.dtype),
                batch_size)
        ],
                                 axis=-1))

    # Broadcast edge structural features to the required batch size.
    # 한글 주석: 에지 구조적 특징을 필요한 배치 크기로 브로드캐스팅합니다.
    grid2mesh_edges_key = grid2mesh_graph.edge_key_by_name("grid2mesh")
    edges = grid2mesh_graph.edges[grid2mesh_edges_key]
    # 에지 특징 업데이트 (배치 차원 추가)
    new_edges = edges._replace(
        features=_add_batch_second_axis(
            edges.features.astype(dummy_mesh_node_features.dtype), batch_size))

    # 업데이트된 노드와 에지로 입력 그래프 구성
    input_graph = self._grid2mesh_graph_structure._replace(
        edges={grid2mesh_edges_key: new_edges},
        nodes={
            "grid_nodes": new_grid_nodes,
            "mesh_nodes": new_mesh_nodes
        })

    # Run the GNN.
    # 한글 주석: GNN을 실행합니다.
    grid2mesh_out = self._grid2mesh_gnn(input_graph, global_norm_conditioning) # GNN 호출
    latent_mesh_nodes = grid2mesh_out.nodes["mesh_nodes"].features # 결과 메쉬 노드 잠재 특징
    latent_grid_nodes = grid2mesh_out.nodes["grid_nodes"].features # 결과 그리드 노드 잠재 특징
    return latent_mesh_nodes, latent_grid_nodes

  def _run_mesh_gnn(self, latent_mesh_nodes: chex.Array,
                    global_norm_conditioning: Optional[chex.Array] = None
                    ) -> chex.Array:
    """Runs the mesh_gnn, extracting updated latent mesh nodes."""
    # 한글 주석: mesh_gnn을 실행하여 업데이트된 잠재 메쉬 노드를 추출합니다.

    # Add the structural edge features of this graph. Note we don't need
    # to add the structural node features, because these are already part of
    # the latent state, via the original Grid2Mesh gnn, however, we need
    # the edge ones, because it is the first time we are seeing this particular
    # set of edges.
    # 한글 주석: 이 그래프의 구조적 에지 특징을 추가합니다. 노드 구조적 특징은 이미 Grid2Mesh gnn을 통해
    # 잠재 상태의 일부이므로 추가할 필요가 없지만, 이 특정 에지 집합은 처음 보므로 에지 특징은 필요합니다.
    batch_size = latent_mesh_nodes.shape[1]

    mesh_graph = self._mesh_graph_structure # 미리 초기화된 메쉬 그래프 구조 사용
    assert mesh_graph is not None
    mesh_edges_key = mesh_graph.edge_key_by_name("mesh")
    edges = mesh_graph.edges[mesh_edges_key]

    # We are assuming here that the mesh gnn uses a single set of edge keys
    # named "mesh" for the edges and that it uses a single set of nodes named
    # "mesh_nodes"
    # 한글 주석: 여기서 메쉬 gnn이 에지에 대해 "mesh"라는 단일 에지 키 집합을 사용하고
    # "mesh_nodes"라는 단일 노드 집합을 사용한다고 가정합니다.
    msg = ("현재 설정은 메쉬 GNN에 한 종류의 에지만을 요구합니다.")
    assert len(mesh_graph.edges) == 1, msg

    # 에지 특징 업데이트 (배치 차원 추가)
    new_edges = edges._replace(
        features=_add_batch_second_axis(
            edges.features.astype(latent_mesh_nodes.dtype), batch_size))

    nodes = mesh_graph.nodes["mesh_nodes"]
    nodes = nodes._replace(features=latent_mesh_nodes) # 입력으로 받은 잠재 메쉬 노드 특징 사용

    # 업데이트된 노드와 에지로 입력 그래프 구성
    input_graph = mesh_graph._replace(
        edges={mesh_edges_key: new_edges}, nodes={"mesh_nodes": nodes})

    # Run the GNN.
    # 한글 주석: GNN을 실행합니다.
    return self._mesh_gnn(input_graph,
                          global_norm_conditioning=global_norm_conditioning # 전역 정규화 조건 전달
                          ).nodes["mesh_nodes"].features # 결과 메쉬 노드 특징 반환

  def _run_mesh2grid_gnn(self,
                         updated_latent_mesh_nodes: chex.Array,
                         latent_grid_nodes: chex.Array,
                         global_norm_conditioning: Optional[chex.Array] = None,
                         ) -> chex.Array:
    """Runs the mesh2grid_gnn, extracting the output grid nodes."""
    # 한글 주석: mesh2grid_gnn을 실행하여 출력 그리드 노드를 추출합니다.

    # Add the structural edge features of this graph. Note we don't need
    # to add the structural node features, because these are already part of
    # the latent state, via the original Grid2Mesh gnn, however, we need
    # the edge ones, because it is the first time we are seeing this particular
    # set of edges.
    # 한글 주석: 이 그래프의 구조적 에지 특징을 추가합니다. (위의 _run_mesh_gnn 주석과 유사)
    batch_size = updated_latent_mesh_nodes.shape[1]

    mesh2grid_graph = self._mesh2grid_graph_structure # 미리 초기화된 그래프 구조 사용
    assert mesh2grid_graph is not None
    mesh_nodes = mesh2grid_graph.nodes["mesh_nodes"]
    grid_nodes = mesh2grid_graph.nodes["grid_nodes"]
    # 입력으로 받은 업데이트된 잠재 메쉬 노드 특징과 잠재 그리드 노드 특징 사용
    new_mesh_nodes = mesh_nodes._replace(features=updated_latent_mesh_nodes)
    new_grid_nodes = grid_nodes._replace(features=latent_grid_nodes)
    mesh2grid_key = mesh2grid_graph.edge_key_by_name("mesh2grid")
    edges = mesh2grid_graph.edges[mesh2grid_key]

    # 에지 특징 업데이트 (배치 차원 추가)
    new_edges = edges._replace(
        features=_add_batch_second_axis(
            edges.features.astype(latent_grid_nodes.dtype), batch_size))

    # 업데이트된 노드와 에지로 입력 그래프 구성
    input_graph = mesh2grid_graph._replace(
        edges={mesh2grid_key: new_edges},
        nodes={
            "mesh_nodes": new_mesh_nodes,
            "grid_nodes": new_grid_nodes
        })

    # Run the GNN.
    # 한글 주석: GNN을 실행합니다.
    output_graph = self._mesh2grid_gnn(input_graph, global_norm_conditioning)
    output_grid_nodes = output_graph.nodes["grid_nodes"].features # 결과 그리드 노드 특징 (최종 출력)

    return output_grid_nodes

  def _inputs_to_grid_node_features_and_norm_conditioning(
      self,
      inputs: xarray.Dataset,
      forcings: xarray.Dataset,
      ) -> Tuple[chex.Array, Optional[chex.Array]]:
    """xarray ->[n_grid_nodes, batch, n_channels], [batch, n_cond channels]."""
    # 한글 주석: xarray 데이터셋을 그리드 노드 특징과 (선택적) 정규화 조건으로 변환합니다.
    # 출력 형태: [그리드_노드_수, 배치, 채널_수], [배치, 조건_채널_수]

    if self._norm_conditioning_features: # 정규화 조건화 특징이 지정된 경우
      # 정규화 조건화에 사용할 입력 추출
      norm_conditioning_inputs = inputs[list(self._norm_conditioning_features)]
      # 원래 입력에서 해당 특징 제거
      inputs = inputs.drop_vars(list(self._norm_conditioning_features))

      if "lat" in norm_conditioning_inputs or "lon" in norm_conditioning_inputs:
        raise ValueError("lat 또는 lon 차원을 가진 특징은 현재 정규화 조건화에 지원되지 않습니다.")
      # 정규화 조건화 입력을 쌓고 전치하여 [배치, 특징] 형태로 만듦
      global_norm_conditioning = xarray_jax.unwrap_data(
          model_utils.dataset_to_stacked(norm_conditioning_inputs,
                                         preserved_dims=("batch",), # 배치 차원 유지
                                         ).transpose("batch", ...)) # 배치 차원을 맨 앞으로
    else: # 정규화 조건화 특징이 지정되지 않은 경우
      global_norm_conditioning = None

    # xarray `Dataset` (batch, time, lat, lon, level, multiple vars)
    # to xarray `DataArray` (batch, lat, lon, channels)
    # 한글 주석: 입력과 강제 변수를 쌓아서 단일 DataArray로 만듭니다.
    stacked_inputs = model_utils.dataset_to_stacked(inputs)
    stacked_forcings = model_utils.dataset_to_stacked(forcings)
    stacked_inputs = xarray.concat( # 입력과 강제 변수를 채널 차원으로 연결
        [stacked_inputs, stacked_forcings], dim="channels")

    # xarray `DataArray` (batch, lat, lon, channels)
    # to single numpy array with shape [lat_lon_node, batch, channels]
    # 한글 주석: (배치, 위도, 경도, 채널) 형태의 DataArray를 (위도*경도_노드, 배치, 채널) 형태의 단일 배열로 변환합니다.
    grid_xarray_lat_lon_leading = model_utils.lat_lon_to_leading_axes( # 위도, 경도 축을 맨 앞으로
        stacked_inputs)
    # ["node", "batch", "features"]
    # 한글 주석: 최종적으로 (노드, 배치, 특징) 형태로 변환합니다.
    grid_node_features = xarray_jax.unwrap(
        grid_xarray_lat_lon_leading.data
    ).reshape((-1,) + grid_xarray_lat_lon_leading.data.shape[2:])
    return grid_node_features, global_norm_conditioning

  def _grid_node_outputs_to_prediction(
      self,
      grid_node_outputs: chex.Array, # [그리드_노드_수, 배치, 출력_수]
      targets_template: xarray.Dataset, # 목표 템플릿 (형태 참조용)
  ) -> xarray.Dataset:
    """[num_grid_nodes, batch, num_outputs] -> xarray."""
    # 한글 주석: [그리드_노드_수, 배치, 출력_수] 형태의 배열을 xarray 데이터셋으로 변환합니다.

    # numpy array with shape [lat_lon_node, batch, channels]
    # to xarray `DataArray` (batch, lat, lon, channels)
    # 한글 주석: (위도*경도_노드, 배치, 채널) 형태의 배열을 (위도, 경도, 배치, 채널) 형태로 재구성합니다.
    assert self._grid_lat is not None and self._grid_lon is not None
    grid_shape = (self._grid_lat.shape[0], self._grid_lon.shape[0]) # 그리드 형태 (위도 수, 경도 수)
    grid_outputs_lat_lon_leading = grid_node_outputs.reshape(
        grid_shape + grid_node_outputs.shape[1:]) # (위도, 경도, 배치, 채널)
    dims = ("lat", "lon", "batch", "channels") # 차원 이름
    # xarray DataArray로 변환
    grid_xarray_lat_lon_leading = xarray_jax.DataArray(
        data=grid_outputs_lat_lon_leading,
        dims=dims)
    # 원래 차원 순서 (배치, 위도, 경도, 채널)로 복원
    grid_xarray = model_utils.restore_leading_axes(grid_xarray_lat_lon_leading)

    # xarray `DataArray` (batch, lat, lon, channels)
    # to xarray `Dataset` (batch, one time step, lat, lon, level, multiple vars)
    # 한글 주석: (배치, 위도, 경도, 채널) 형태의 DataArray를 원래의 다차원 데이터셋 형태로 변환합니다.
    return model_utils.stacked_to_dataset(
        grid_xarray.variable, targets_template)


def _add_batch_second_axis(data, batch_size):
  # data [leading_dim, trailing_dim]
  # 한글 주석: 데이터의 두 번째 축으로 배치 차원을 추가합니다. 입력 데이터는 [선행_차원, 후행_차원] 형태여야 합니다.
  assert data.ndim == 2 # 입력 데이터가 2차원인지 확인
  ones = jnp.ones([batch_size, 1], dtype=data.dtype) # 배치 크기의 1로 채워진 배열 생성
  return data[:, None] * ones  # [leading_dim, batch, trailing_dim] # 브로드캐스팅을 이용해 배치 차원 추가


def _get_max_edge_distance(mesh):
  # N.B.To make sure ordering is preserved, any changes to faces_to_edges here
  # should be reflected in the other 2 calls to faces_to_edges in this file.
  # 한글 주석: 메쉬 내의 최대 에지 거리를 계산합니다.
  # 참고: 순서 보존을 위해 여기서 faces_to_edges에 대한 변경 사항은 이 파일의 다른 2개 faces_to_edges 호출에도 반영되어야 합니다.
  senders, receivers = icosahedral_mesh.faces_to_edges(mesh.faces) # 면에서 에지 추출
  # 각 에지의 유클리드 거리 계산
  edge_distances = np.linalg.norm(
      mesh.vertices[senders] - mesh.vertices[receivers], axis=-1)
  return edge_distances.max() # 최대 거리 반환


def _permute_mesh_to_banded(mesh):
  """Permutes the mesh nodes such that adjacency matrix has banded structure."""
  # 한글 주석: 인접 행렬이 띠 구조(banded structure)를 갖도록 메쉬 노드를 순열합니다.
  # Build adjacency matrix.
  # 한글 주석: 인접 행렬을 구축합니다.
  # N.B.To make sure ordering is preserved, any changes to faces_to_edges here
  # should be reflected in the other 2 calls to faces_to_edges in this file.
  # 한글 주석: (위와 동일한 참고 사항)
  senders, receivers = icosahedral_mesh.faces_to_edges(mesh.faces)
  num_mesh_nodes = mesh.vertices.shape[0] # 메쉬 노드 수
  adj_mat = sparse.csr_matrix((num_mesh_nodes, num_mesh_nodes)) # 희소 CSR 행렬 생성
  adj_mat[senders, receivers] = 1 # 연결된 노드에 1 할당
  # Permutation to banded (this algorithm is deterministic, a given sparse
  # adjacency matrix will yield the same permutation every time this is run).
  # 한글 주석: 띠 구조로 순열합니다 (이 알고리즘은 결정론적이므로 주어진 희소 인접 행렬은 항상 동일한 순열을 생성합니다).
  # Reverse Cuthill-McKee 알고리즘 사용
  mesh_permutation = sparse.csgraph.reverse_cuthill_mckee(
      adj_mat, symmetric_mode=True # 대칭 모드 사용
  )
  # 원래 인덱스와 순열된 인덱스 간의 매핑 생성
  vertex_permutation_map = {j: i for i, j in enumerate(mesh_permutation)}
  # 면 정보를 새로운 순열에 맞게 변환하는 함수
  permute_func = np.vectorize(lambda x: vertex_permutation_map[x])
  # 순열된 정점과 면 정보로 새로운 TriangularMesh 객체 생성
  return icosahedral_mesh.TriangularMesh(
      vertices=mesh.vertices[mesh_permutation], faces=permute_func(mesh.faces)
  )
