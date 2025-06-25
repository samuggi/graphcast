# Copyright 2023 DeepMind Technologies Limited.
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
"""A predictor that runs multiple graph neural networks on mesh data.
# 한글 주석: 메쉬 데이터에서 여러 그래프 신경망을 실행하는 예측기입니다.

It learns to interpolate between the grid and the mesh nodes, with the loss
and the rollouts ultimately computed at the grid level.
# 한글 주석: 그리드와 메쉬 노드 간의 보간을 학습하며, 손실 및 롤아웃은 궁극적으로 그리드 수준에서 계산됩니다.

It uses ideas similar to those in Keisler (2022):
# 한글 주석: Keisler (2022)의 아이디어와 유사한 아이디어를 사용합니다:

Reference:
  https://arxiv.org/pdf/2202.07575.pdf

It assumes data across time and level is stacked, and operates only operates in
a 2D mesh over latitudes and longitudes.
# 한글 주석: 시간과 레벨에 걸친 데이터가 쌓여 있다고 가정하며, 위도와 경도에 대한 2D 메쉬에서만 작동합니다.
"""

from typing import Any, Callable, Mapping, Optional # 한글 주석: 타입 힌트를 위한 모듈 임포트

import chex # JAX 및 NumPy를 위한 유틸리티 라이브러리
from graphcast import deep_typed_graph_net # 심층 타입 그래프 신경망 모듈
from graphcast import grid_mesh_connectivity # 그리드-메쉬 연결성 관련 모듈
from graphcast import icosahedral_mesh # 정이십면체 메쉬 관련 모듈
from graphcast import losses # 손실 함수 모듈
from graphcast import model_utils # 모델 유틸리티 함수
from graphcast import predictor_base # 예측기 기본 클래스
from graphcast import typed_graph # 타입 그래프 관련 모듈
from graphcast import xarray_jax # Xarray와 JAX 연동 유틸리티
import jax.numpy as jnp
import jraph # Jraph: JAX를 위한 그래프 신경망 라이브러리
import numpy as np
import xarray

Kwargs = Mapping[str, Any] # 키워드 인자 타입 정의

GNN = Callable[[jraph.GraphsTuple], jraph.GraphsTuple] # GNN 함수 타입 정의


# https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5
PRESSURE_LEVELS_ERA5_37 = ( # ERA5 데이터셋의 37개 기압 수준 (hPa)
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300,
    350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900,
    925, 950, 975, 1000)

# https://www.ecmwf.int/en/forecasts/datasets/set-i
PRESSURE_LEVELS_HRES_25 = ( # HRES 데이터셋의 25개 기압 수준 (hPa)
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600,
    700, 800, 850, 900, 925, 950, 1000)

# https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020MS002203
PRESSURE_LEVELS_WEATHERBENCH_13 = ( # WeatherBench 데이터셋의 13개 기압 수준 (hPa)
    50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)

PRESSURE_LEVELS = { # 사용 가능한 기압 수준 세트 딕셔너리
    13: PRESSURE_LEVELS_WEATHERBENCH_13,
    25: PRESSURE_LEVELS_HRES_25,
    37: PRESSURE_LEVELS_ERA5_37,
}

# The list of all possible atmospheric variables. Taken from:
# https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation#ERA5:datadocumentation-Table9
# 한글 주석: 가능한 모든 대기 변수 목록입니다. 출처: ECMWF ERA5 문서
ALL_ATMOSPHERIC_VARS = (
    "potential_vorticity", # 잠재 와도
    "specific_rain_water_content", # 특정 강우량
    "specific_snow_water_content", # 특정 강설량
    "geopotential", # 지오포텐셜
    "temperature", # 온도
    "u_component_of_wind", # 바람의 U 성분
    "v_component_of_wind", # 바람의 V 성분
    "specific_humidity", # 비습
    "vertical_velocity", # 연직 속도
    "vorticity", # 와도
    "divergence", # 발산
    "relative_humidity", # 상대 습도
    "ozone_mass_mixing_ratio", # 오존 질량 혼합비
    "specific_cloud_liquid_water_content", # 특정 구름 액체 물 함량
    "specific_cloud_ice_water_content", # 특정 구름 얼음 물 함량
    "fraction_of_cloud_cover", # 운량
)

TARGET_SURFACE_VARS = ( # 목표 지표면 변수
    "2m_temperature", # 2m 온도
    "mean_sea_level_pressure", # 평균 해수면 기압
    "10m_v_component_of_wind", # 10m 바람의 V 성분
    "10m_u_component_of_wind", # 10m 바람의 U 성분
    "total_precipitation_6hr", # 6시간 총 강수량
)
TARGET_SURFACE_NO_PRECIP_VARS = ( # 강수량을 제외한 목표 지표면 변수
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_v_component_of_wind",
    "10m_u_component_of_wind",
)
TARGET_ATMOSPHERIC_VARS = ( # 목표 대기 변수
    "temperature",
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "specific_humidity",
)
TARGET_ATMOSPHERIC_NO_W_VARS = ( # 연직 속도를 제외한 목표 대기 변수
    "temperature",
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
)
EXTERNAL_FORCING_VARS = ( # 외부 강제 변수
    "toa_incident_solar_radiation", # 대기 상층 입사 태양 복사
)
GENERATED_FORCING_VARS = ( # 생성된 강제 변수 (시간적 특징)
    "year_progress_sin", # 연 진행률 (sin)
    "year_progress_cos", # 연 진행률 (cos)
    "day_progress_sin", # 일 진행률 (sin)
    "day_progress_cos", # 일 진행률 (cos)
)
FORCING_VARS = EXTERNAL_FORCING_VARS + GENERATED_FORCING_VARS # 모든 강제 변수
STATIC_VARS = ( # 정적 변수 (시간에 따라 변하지 않음)
    "geopotential_at_surface", # 지표면 지오포텐셜
    "land_sea_mask", # 육지-바다 마스크
)


@chex.dataclass(frozen=True, eq=True)
class TaskConfig:
  """Defines inputs and targets on which a model is trained and/or evaluated."""
  # 한글 주석: 모델이 학습 및/또는 평가되는 입력 및 목표를 정의합니다.
  input_variables: tuple[str, ...] # 입력 변수 목록
  # Target variables which the model is expected to predict.
  # 한글 주석: 모델이 예측할 것으로 예상되는 목표 변수입니다.
  target_variables: tuple[str, ...] # 목표 변수 목록
  forcing_variables: tuple[str, ...] # 강제 변수 목록
  pressure_levels: tuple[int, ...] # 기압 수준 목록
  input_duration: str # 입력 기간 (예: "12h")

# 기본 작업 구성 (ERA5 37 레벨 기준)
TASK = TaskConfig(
    input_variables=( # 모든 사용 가능한 변수를 입력으로 사용
        TARGET_SURFACE_VARS + TARGET_ATMOSPHERIC_VARS + FORCING_VARS +
        STATIC_VARS),
    target_variables=TARGET_SURFACE_VARS + TARGET_ATMOSPHERIC_VARS, # 지표면 및 대기 변수를 목표로 함
    forcing_variables=FORCING_VARS, # 모든 강제 변수 사용
    pressure_levels=PRESSURE_LEVELS_ERA5_37, # ERA5 37개 기압 수준 사용
    input_duration="12h", # 12시간 입력 기간
)
# WeatherBench 13 레벨 기준 작업 구성
TASK_13 = TaskConfig(
    input_variables=(
        TARGET_SURFACE_VARS + TARGET_ATMOSPHERIC_VARS + FORCING_VARS +
        STATIC_VARS),
    target_variables=TARGET_SURFACE_VARS + TARGET_ATMOSPHERIC_VARS,
    forcing_variables=FORCING_VARS,
    pressure_levels=PRESSURE_LEVELS_WEATHERBENCH_13, # WeatherBench 13개 기압 수준 사용
    input_duration="12h",
)
# WeatherBench 13 레벨, 강수량 출력 작업 구성 (입력에서는 강수량 제외)
TASK_13_PRECIP_OUT = TaskConfig(
    input_variables=(
        TARGET_SURFACE_NO_PRECIP_VARS + TARGET_ATMOSPHERIC_VARS + FORCING_VARS + # 입력 변수에서 강수량 관련 변수 제외
        STATIC_VARS),
    target_variables=TARGET_SURFACE_VARS + TARGET_ATMOSPHERIC_VARS, # 목표 변수에는 강수량 포함
    forcing_variables=FORCING_VARS,
    pressure_levels=PRESSURE_LEVELS_WEATHERBENCH_13,
    input_duration="12h",
)


@chex.dataclass(frozen=True, eq=True)
class ModelConfig:
  """Defines the architecture of the GraphCast neural network architecture.
  # 한글 주석: GraphCast 신경망 아키텍처의 구조를 정의합니다.

  Properties:
  # 한글 주석: 속성:
    resolution: The resolution of the data, in degrees (e.g. 0.25 or 1.0).
    # 한글 주석: resolution: 데이터의 해상도 (도 단위, 예: 0.25 또는 1.0).
    mesh_size: How many refinements to do on the multi-mesh.
    # 한글 주석: mesh_size: 다중 메쉬에 적용할 개선 횟수입니다. (정이십면체 분할 횟수)
    gnn_msg_steps: How many Graph Network message passing steps to do.
    # 한글 주석: gnn_msg_steps: 수행할 그래프 신경망 메시지 전달 단계 수입니다. (프로세서 GNN의 단계 수)
    latent_size: How many latent features to include in the various MLPs.
    # 한글 주석: latent_size: 다양한 MLP에 포함할 잠재 특징의 수입니다. (임베딩 및 내부 표현 차원)
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
    mesh2grid_edge_normalization_factor: Allows explicitly controlling edge
        normalization for mesh2grid edges. If None, defaults to max edge length.
        This supports using pre-trained model weights with a different graph
        structure to what it was trained on.
    # 한글 주석: mesh2grid_edge_normalization_factor: mesh2grid 에지에 대한 에지 정규화를 명시적으로 제어할 수 있습니다. None이면 최대 에지 길이로 기본 설정됩니다.
    # 이는 학습된 그래프 구조와 다른 그래프 구조로 사전 학습된 모델 가중치를 사용하는 것을 지원합니다.
  """
  resolution: float # 해상도 (도 단위)
  mesh_size: int # 메쉬 분할 횟수
  latent_size: int # 잠재 공간 차원
  gnn_msg_steps: int # GNN 메시지 전달 단계 수
  hidden_layers: int # MLP 은닉층 수
  radius_query_fraction_edge_length: float # Grid2Mesh 연결 반경 결정을 위한 가장 긴 에지 길이 비율
  mesh2grid_edge_normalization_factor: Optional[float] = None # Mesh2Grid 에지 정규화 인자 (선택 사항)


@chex.dataclass(frozen=True, eq=True)
class CheckPoint:
  # 한글 주석: 모델 체크포인트 정보를 저장하는 데이터 클래스입니다.
  params: dict[str, Any] # 모델 파라미터 (일반적으로 Haiku 파라미터 트리)
  model_config: ModelConfig # 모델 아키텍처 설정
  task_config: TaskConfig # 작업(데이터) 설정
  description: str # 체크포인트 설명
  license: str # 라이선스 정보


class GraphCast(predictor_base.Predictor):
  """GraphCast Predictor.
  # 한글 주석: GraphCast 예측기 클래스입니다.

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
    nodes is connected to 3 nodes of the mesh triangular face that contains
    the grid points. The mesh2grid_gnn will operate in this graph. It will
    process the updated latent state of the mesh nodes, and the latent state
    of the grid nodes, to produce the final output for the grid nodes.
  # 한글 주석: * Mesh2Grid 그래프: 모든 노드를 포함하는 그래프입니다. 이 그래프는 각 그리드 노드가 그리드 지점을 포함하는 메쉬 삼각형 면의 3개 노드에 연결되도록 메쉬 노드에서 그리드 노드로 향하는 에지를 가진 엄격한 이분 그래프입니다. mesh2grid_gnn이 이 그래프에서 작동합니다. 메쉬 노드의 업데이트된 잠재 상태와 그리드 노드의 잠재 상태를 처리하여 그리드 노드에 대한 최종 출력을 생성합니다.

  The model is built on top of `TypedGraph`s so the different types of nodes and
  edges can be stored and treated separately.
  # 한글 주석: 이 모델은 `TypedGraph` 위에 구축되어 다양한 유형의 노드와 에지를 별도로 저장하고 처리할 수 있습니다.
  """

  def __init__(self, model_config: ModelConfig, task_config: TaskConfig):
    """Initializes the predictor."""
    # 한글 주석: 예측기를 초기화합니다.
    # 공간 특징 생성을 위한 키워드 인수들을 저장합니다.
    self._spatial_features_kwargs = dict(
        add_node_positions=False, # 노드 (x,y,z) 위치 추가 안 함
        add_node_latitude=True, # 노드 위도 특징 추가
        add_node_longitude=True, # 노드 경도 특징 추가
        add_relative_positions=True, # 에지에 대한 상대 위치 특징 추가
        relative_longitude_local_coordinates=True, # 상대 경도를 로컬 좌표계로 변환
        relative_latitude_local_coordinates=True, # 상대 위도를 로컬 좌표계로 변환
    )

    # Specification of the multimesh.
    # 한글 주석: 다중 메쉬(multi-mesh)를 정의합니다. 정이십면체를 분할하여 계층적인 메쉬를 생성합니다.
    self._meshes = (
        icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(
            splits=model_config.mesh_size)) # 메쉬 분할 횟수는 모델 설정에서 가져옴

    # Encoder, which moves data from the grid to the mesh with a single message
    # passing step.
    # 한글 주석: 인코더 GNN입니다. 그리드 데이터를 메쉬로 옮기는 역할을 하며, 단일 메시지 전달 단계를 수행합니다.
    self._grid2mesh_gnn = deep_typed_graph_net.DeepTypedGraphNet(
        embed_nodes=True,  # Embed raw features of the grid and mesh nodes. # 그리드 및 메쉬 노드의 원시 특징을 임베딩합니다.
        embed_edges=True,  # Embed raw features of the grid2mesh edges. # grid2mesh 에지의 원시 특징을 임베딩합니다.
        edge_latent_size=dict(grid2mesh=model_config.latent_size), # grid2mesh 에지의 잠재 공간 크기
        node_latent_size=dict( # 노드 타입별 잠재 공간 크기
            mesh_nodes=model_config.latent_size, # 메쉬 노드
            grid_nodes=model_config.latent_size), # 그리드 노드
        mlp_hidden_size=model_config.latent_size, # MLP 은닉층 크기
        mlp_num_hidden_layers=model_config.hidden_layers, # MLP 은닉층 수
        num_message_passing_steps=1, # 메시지 전달 단계 수 (인코더는 1회)
        use_layer_norm=True, # 레이어 정규화 사용
        include_sent_messages_in_node_update=False, # 노드 업데이트 시 보낸 메시지 미포함
        activation="swish", # 활성화 함수
        f32_aggregation=True, # 에지 집계 시 float32 사용 (정밀도 향상)
        aggregate_normalization=None, # 집계 정규화 없음
        name="grid2mesh_gnn", # 모듈 이름
    )

    # Processor, which performs message passing on the multi-mesh.
    # 한글 주석: 프로세서 GNN입니다. 다중 메쉬 상에서 메시지 전달을 수행합니다.
    self._mesh_gnn = deep_typed_graph_net.DeepTypedGraphNet(
        embed_nodes=False,  # Node features already embdded by previous layers. # 노드 특징은 이미 이전 계층에서 임베딩됨
        embed_edges=True,  # Embed raw features of the multi-mesh edges. # 다중 메쉬 에지의 원시 특징을 임베딩합니다.
        node_latent_size=dict(mesh_nodes=model_config.latent_size), # 메쉬 노드의 잠재 공간 크기
        edge_latent_size=dict(mesh=model_config.latent_size), # 메쉬 에지의 잠재 공간 크기
        mlp_hidden_size=model_config.latent_size,
        mlp_num_hidden_layers=model_config.hidden_layers,
        num_message_passing_steps=model_config.gnn_msg_steps, # 설정된 메시지 전달 단계 수만큼 반복
        use_layer_norm=True,
        include_sent_messages_in_node_update=False,
        activation="swish",
        f32_aggregation=False, # 여기서는 float32 집계 사용 안 함
        name="mesh_gnn",
    )

    # 출력 변수의 총 개수를 계산합니다.
    num_surface_vars = len( # 지표면 변수 개수
        set(task_config.target_variables) - set(ALL_ATMOSPHERIC_VARS))
    num_atmospheric_vars = len( # 대기 변수 개수
        set(task_config.target_variables) & set(ALL_ATMOSPHERIC_VARS))
    num_outputs = (num_surface_vars + # 총 출력 수 = 지표면 변수 수 + (기압 수준 수 * 대기 변수 수)
                   len(task_config.pressure_levels) * num_atmospheric_vars)

    # Decoder, which moves data from the mesh back into the grid with a single
    # message passing step.
    # 한글 주석: 디코더 GNN입니다. 메쉬에서 그리드로 데이터를 다시 옮기는 역할을 하며, 단일 메시지 전달 단계를 수행합니다.
    self._mesh2grid_gnn = deep_typed_graph_net.DeepTypedGraphNet(
        # Require a specific node dimensionaly for the grid node outputs.
        # 한글 주석: 그리드 노드 출력에 대해 특정 노드 차원을 요구합니다.
        node_output_size=dict(grid_nodes=num_outputs), # 그리드 노드의 출력 크기를 위에서 계산한 num_outputs로 설정
        embed_nodes=False,  # Node features already embdded by previous layers.
        embed_edges=True,  # Embed raw features of the mesh2grid edges.
        edge_latent_size=dict(mesh2grid=model_config.latent_size),
        node_latent_size=dict(
            mesh_nodes=model_config.latent_size,
            grid_nodes=model_config.latent_size),
        mlp_hidden_size=model_config.latent_size,
        mlp_num_hidden_layers=model_config.hidden_layers,
        num_message_passing_steps=1, # 디코더도 1회 메시지 전달
        use_layer_norm=True,
        include_sent_messages_in_node_update=False,
        activation="swish",
        f32_aggregation=False,
        name="mesh2grid_gnn",
    )

    # Obtain the query radius in absolute units for the unit-sphere for the
    # grid2mesh model, by rescaling the `radius_query_fraction_edge_length`.
    # 한글 주석: grid2mesh 모델에 대한 단위 구의 절대 단위 쿼리 반경을 `radius_query_fraction_edge_length`를 재조정하여 얻습니다.
    self._query_radius = (_get_max_edge_distance(self._finest_mesh) # 가장 미세한 메쉬의 최대 에지 거리
                          * model_config.radius_query_fraction_edge_length) # 설정된 비율 곱하기
    self._mesh2grid_edge_normalization_factor = ( # mesh2grid 에지 정규화 인자
        model_config.mesh2grid_edge_normalization_factor
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
    # 한글 주석: "_init_{grid2mesh,processor,mesh2grid}_graph": 그래프 구조들을 저장할 변수들입니다.
    self._grid2mesh_graph_structure = None
    self._mesh_graph_structure = None
    self._mesh2grid_graph_structure = None

  @property
  def _finest_mesh(self):
    # 한글 주석: 계층적 메쉬 중 가장 해상도가 높은 (가장 미세한) 메쉬를 반환합니다.
    return self._meshes[-1]

  def __call__(self,
               inputs: xarray.Dataset,
               targets_template: xarray.Dataset,
               forcings: xarray.Dataset,
               is_training: bool = False, # 학습 중인지 여부 (현재 사용되지 않음)
               ) -> xarray.Dataset:
    # 한글 주석: 모델의 순전파를 수행합니다.
    self._maybe_init(inputs) # 필요시 모델 및 그래프 구조 초기화

    # Convert all input data into flat vectors for each of the grid nodes.
    # xarray (batch, time, lat, lon, level, multiple vars, forcings)
    # -> [num_grid_nodes, batch, num_channels]
    # 한글 주석: 모든 입력 데이터를 각 그리드 노드에 대한 평탄한 벡터로 변환합니다.
    grid_node_features = self._inputs_to_grid_node_features(inputs, forcings)

    # Transfer data for the grid to the mesh,
    # [num_mesh_nodes, batch, latent_size], [num_grid_nodes, batch, latent_size]
    # 한글 주석: Grid2Mesh GNN을 실행하여 그리드 데이터를 메쉬로 전달하고, 메쉬와 그리드 노드의 잠재 표현을 얻습니다.
    (latent_mesh_nodes, latent_grid_nodes
     ) = self._run_grid2mesh_gnn(grid_node_features)

    # Run message passing in the multimesh.
    # [num_mesh_nodes, batch, latent_size]
    # 한글 주석: 다중 메쉬에서 메시지 전달을 실행하여 메쉬 노드의 잠재 표현을 업데이트합니다.
    updated_latent_mesh_nodes = self._run_mesh_gnn(latent_mesh_nodes)

    # Transfer data frome the mesh to the grid.
    # [num_grid_nodes, batch, output_size]
    # 한글 주석: Mesh2Grid GNN을 실행하여 메쉬에서 그리드로 데이터를 전달하고, 그리드 노드의 최종 출력을 얻습니다.
    output_grid_nodes = self._run_mesh2grid_gnn(
        updated_latent_mesh_nodes, latent_grid_nodes)

    # Conver output flat vectors for the grid nodes to the format of the output.
    # [num_grid_nodes, batch, output_size] ->
    # xarray (batch, one time step, lat, lon, level, multiple vars)
    # 한글 주석: 그리드 노드에 대한 출력 평탄 벡터를 원래 xarray 데이터셋 형식으로 변환합니다.
    return self._grid_node_outputs_to_prediction(
        output_grid_nodes, targets_template)

  def loss_and_predictions(  # pytype: disable=signature-mismatch  # jax-ndarray
      self,
      inputs: xarray.Dataset,
      targets: xarray.Dataset,
      forcings: xarray.Dataset,
      ) -> tuple[predictor_base.LossAndDiagnostics, xarray.Dataset]:
    # 한글 주석: 손실과 예측을 함께 계산하여 반환합니다.
    # Forward pass.
    # 한글 주석: 순전파를 수행하여 예측값을 얻습니다.
    predictions = self(
        inputs, targets_template=targets, forcings=forcings, is_training=True)
    # Compute loss.
    # 한글 주석: 예측값과 실제 목표값 사이의 손실을 계산합니다.
    loss = losses.weighted_mse_per_level( # 가중 평균 제곱 오차 (레벨별)
        predictions, targets,
        per_variable_weights={ # 변수별 가중치 설정
            # Any variables not specified here are weighted as 1.0.
            # 한글 주석: 여기에 지정되지 않은 변수는 1.0으로 가중치가 부여됩니다.
            # A single-level variable, but an important headline variable
            # and also one which we have struggled to get good performance
            # on at short lead times, so leaving it weighted at 1.0, equal
            # to the multi-level variables:
            # 한글 주석: 단일 레벨 변수이지만 중요한 주요 변수이며 짧은 리드 타임에서 좋은 성능을 얻기 어려웠으므로
            # 다중 레벨 변수와 동일하게 1.0으로 가중치를 유지합니다.
            "2m_temperature": 1.0,
            # New single-level variables, which we don't weight too highly
            # to avoid hurting performance on other variables.
            # 한글 주석: 다른 변수의 성능을 저해하지 않도록 너무 높게 가중치를 부여하지 않는 새로운 단일 레벨 변수입니다.
            "10m_u_component_of_wind": 0.1,
            "10m_v_component_of_wind": 0.1,
            "mean_sea_level_pressure": 0.1,
            "total_precipitation_6hr": 0.1,
        })
    return loss, predictions  # pytype: disable=bad-return-type  # jax-ndarray

  def loss(  # pytype: disable=signature-mismatch  # jax-ndarray
      self,
      inputs: xarray.Dataset,
      targets: xarray.Dataset,
      forcings: xarray.Dataset,
      ) -> predictor_base.LossAndDiagnostics:
    # 한글 주석: 손실만 계산하여 반환합니다.
    loss, _ = self.loss_and_predictions(inputs, targets, forcings)
    return loss  # pytype: disable=bad-return-type  # jax-ndarray

  def _maybe_init(self, sample_inputs: xarray.Dataset):
    """Inits everything that has a dependency on the input coordinates."""
    # 한글 주석: 입력 좌표에 의존하는 모든 것을 초기화합니다 (필요한 경우).
    if not self._initialized: # 아직 초기화되지 않았다면
      self._init_mesh_properties() # 메쉬 속성 초기화
      self._init_grid_properties( # 그리드 속성 초기화 (샘플 입력의 위도/경도 사용)
          grid_lat=sample_inputs.lat, grid_lon=sample_inputs.lon)
      self._grid2mesh_graph_structure = self._init_grid2mesh_graph() # Grid2Mesh 그래프 구조 초기화
      self._mesh_graph_structure = self._init_mesh_graph() # 메쉬 그래프 구조 초기화
      self._mesh2grid_graph_structure = self._init_mesh2grid_graph() # Mesh2Grid 그래프 구조 초기화

      self._initialized = True # 초기화 완료 플래그 설정

  def _init_mesh_properties(self):
    """Inits static properties that have to do with mesh nodes."""
    # 한글 주석: 메쉬 노드와 관련된 정적 속성을 초기화합니다.
    self._num_mesh_nodes = self._finest_mesh.vertices.shape[0] # 가장 미세한 메쉬의 정점 수
    # 메쉬 정점의 데카르트 좌표를 구면 좌표(phi, theta)로 변환합니다.
    mesh_phi, mesh_theta = model_utils.cartesian_to_spherical(
        self._finest_mesh.vertices[:, 0],
        self._finest_mesh.vertices[:, 1],
        self._finest_mesh.vertices[:, 2])
    # 구면 좌표를 위도(latitude)와 경도(longitude)로 변환합니다.
    (
        mesh_nodes_lat,
        mesh_nodes_lon,
    ) = model_utils.spherical_to_lat_lon(
        phi=mesh_phi, theta=mesh_theta)
    # Convert to f32 to ensure the lat/lon features aren't in f64.
    # 한글 주석: 위도/경도 특징이 float64가 아닌 float32인지 확인하기 위해 변환합니다.
    self._mesh_nodes_lat = mesh_nodes_lat.astype(np.float32)
    self._mesh_nodes_lon = mesh_nodes_lon.astype(np.float32)

  def _init_grid_properties(self, grid_lat: np.ndarray, grid_lon: np.ndarray):
    """Inits static properties that have to do with grid nodes."""
    # 한글 주석: 그리드 노드와 관련된 정적 속성을 초기화합니다.
    self._grid_lat = grid_lat.astype(np.float32) # 그리드 위도
    self._grid_lon = grid_lon.astype(np.float32) # 그리드 경도
    # Initialized the counters.
    # 한글 주석: 카운터를 초기화합니다.
    self._num_grid_nodes = grid_lat.shape[0] * grid_lon.shape[0] # 총 그리드 노드 수

    # Initialize lat and lon for the grid.
    # 한글 주석: 그리드의 각 노드에 대한 위도와 경도를 초기화합니다.
    grid_nodes_lon, grid_nodes_lat = np.meshgrid(grid_lon, grid_lat) # 경도, 위도로부터 그리드 생성
    self._grid_nodes_lon = grid_nodes_lon.reshape([-1]).astype(np.float32) # 1D 배열로 변환
    self._grid_nodes_lat = grid_nodes_lat.reshape([-1]).astype(np.float32) # 1D 배열로 변환

  def _init_grid2mesh_graph(self) -> typed_graph.TypedGraph:
    """Build Grid2Mesh graph."""
    # 한글 주석: Grid2Mesh 그래프를 구축합니다. 이 그래프는 그리드 노드에서 메쉬 노드로 정보를 전달합니다.

    # Create some edges according to distance between mesh and grid nodes.
    # 한글 주석: 메쉬 노드와 그리드 노드 간의 거리에 따라 에지를 생성합니다.
    assert self._grid_lat is not None and self._grid_lon is not None
    # 반경 쿼리를 사용하여 특정 반경 내에 있는 그리드-메쉬 노드 쌍을 찾습니다.
    (grid_indices, mesh_indices) = grid_mesh_connectivity.radius_query_indices(
        grid_latitude=self._grid_lat,
        grid_longitude=self._grid_lon,
        mesh=self._finest_mesh, # 가장 미세한 메쉬 사용
        radius=self._query_radius) # 미리 계산된 쿼리 반경 사용

    # Edges sending info from grid to mesh.
    # 한글 주석: 그리드에서 메쉬로 정보를 보내는 에지입니다.
    senders = grid_indices # 송신 노드 (그리드 인덱스)
    receivers = mesh_indices # 수신 노드 (메쉬 인덱스)

    # Precompute structural node and edge features according to config options.
    # Structural features are those that depend on the fixed values of the
    # latitude and longitudes of the nodes.
    # 한글 주석: 설정 옵션에 따라 구조적 노드 및 에지 특징을 미리 계산합니다.
    # 구조적 특징은 노드의 위도 및 경도의 고정된 값에 따라 달라집니다.
    (senders_node_features, receivers_node_features,
     edge_features) = model_utils.get_bipartite_graph_spatial_features( # 이분 그래프 공간 특징 계산
         senders_node_lat=self._grid_nodes_lat, # 송신 노드(그리드) 위도
         senders_node_lon=self._grid_nodes_lon, # 송신 노드(그리드) 경도
         receivers_node_lat=self._mesh_nodes_lat, # 수신 노드(메쉬) 위도
         receivers_node_lon=self._mesh_nodes_lon, # 수신 노드(메쉬) 경도
         senders=senders,
         receivers=receivers,
         edge_normalization_factor=None, # 에지 정규화 인자 (여기서는 사용 안 함)
         **self._spatial_features_kwargs, # 공간 특징 생성 관련 인수들
     )

    n_grid_node = np.array([self._num_grid_nodes]) # 그리드 노드 총 수
    n_mesh_node = np.array([self._num_mesh_nodes]) # 메쉬 노드 총 수
    n_edge = np.array([mesh_indices.shape[0]]) # 생성된 에지 총 수
    # 그리드 노드 집합(NodeSet) 생성
    grid_node_set = typed_graph.NodeSet(
        n_node=n_grid_node, features=senders_node_features) # 송신자(그리드)의 특징 사용
    # 메쉬 노드 집합(NodeSet) 생성
    mesh_node_set = typed_graph.NodeSet(
        n_node=n_mesh_node, features=receivers_node_features) # 수신자(메쉬)의 특징 사용
    # 에지 집합(EdgeSet) 생성
    edge_set = typed_graph.EdgeSet(
        n_edge=n_edge,
        indices=typed_graph.EdgesIndices(senders=senders, receivers=receivers),
        features=edge_features)
    nodes = {"grid_nodes": grid_node_set, "mesh_nodes": mesh_node_set} # 노드 딕셔너리
    edges = { # 에지 딕셔너리
        typed_graph.EdgeSetKey("grid2mesh", ("grid_nodes", "mesh_nodes")): # 에지 타입 키 ("grid2mesh", 소스 타입, 타겟 타입)
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
    # 한글 주석: 메쉬 그래프를 구축합니다. 이 그래프는 메쉬 노드 간의 상호작용을 나타냅니다.
    merged_mesh = icosahedral_mesh.merge_meshes(self._meshes) # 계층적 메쉬들을 병합 (여기서는 단일 _finest_mesh와 동일할 수 있음)

    # Work simply on the mesh edges.
    # 한글 주석: 메쉬 에지에 대해 간단히 작업합니다.
    senders, receivers = icosahedral_mesh.faces_to_edges(merged_mesh.faces) # 메쉬 면에서 에지(송신자, 수신자) 추출

    # Precompute structural node and edge features according to config options.
    # Structural features are those that depend on the fixed values of the
    # latitude and longitudes of the nodes.
    # 한글 주석: (위와 동일한 설명)
    assert self._mesh_nodes_lat is not None and self._mesh_nodes_lon is not None
    node_features, edge_features = model_utils.get_graph_spatial_features( # 그래프 공간 특징 계산
        node_lat=self._mesh_nodes_lat, # 메쉬 노드 위도
        node_lon=self._mesh_nodes_lon, # 메쉬 노드 경도
        senders=senders,
        receivers=receivers,
        **self._spatial_features_kwargs,
    )

    n_mesh_node = np.array([self._num_mesh_nodes])
    n_edge = np.array([senders.shape[0]])
    assert n_mesh_node == len(node_features) # 노드 수와 특징 배열 길이 일치 확인
    mesh_node_set = typed_graph.NodeSet(
        n_node=n_mesh_node, features=node_features)
    edge_set = typed_graph.EdgeSet(
        n_edge=n_edge,
        indices=typed_graph.EdgesIndices(senders=senders, receivers=receivers),
        features=edge_features)
    nodes = {"mesh_nodes": mesh_node_set} # 메쉬 노드만 포함
    edges = {
        typed_graph.EdgeSetKey("mesh", ("mesh_nodes", "mesh_nodes")): edge_set # "mesh" 타입 에지 (메쉬 노드 간)
    }
    mesh_graph = typed_graph.TypedGraph(
        context=typed_graph.Context(n_graph=np.array([1]), features=()),
        nodes=nodes,
        edges=edges)

    return mesh_graph

  def _init_mesh2grid_graph(self) -> typed_graph.TypedGraph:
    """Build Mesh2Grid graph."""
    # 한글 주석: Mesh2Grid 그래프를 구축합니다. 이 그래프는 메쉬 노드에서 그리드 노드로 정보를 전달합니다.

    # Create some edges according to how the grid nodes are contained by
    # mesh triangles.
    # 한글 주석: 그리드 노드가 메쉬 삼각형에 포함되는 방식에 따라 에지를 생성합니다.
    # 각 그리드 노드를 포함하는 메쉬 삼각형의 꼭짓점(메쉬 노드)들을 찾습니다.
    (grid_indices, # 그리드 노드 인덱스
     mesh_indices) = grid_mesh_connectivity.in_mesh_triangle_indices( # 메쉬 삼각형 내 그리드 노드 인덱스 찾기
         grid_latitude=self._grid_lat,
         grid_longitude=self._grid_lon,
         mesh=self._finest_mesh) # 가장 미세한 메쉬 사용

    # Edges sending info from mesh to grid.
    # 한글 주석: 메쉬에서 그리드로 정보를 보내는 에지입니다.
    senders = mesh_indices # 송신 노드 (메쉬 인덱스)
    receivers = grid_indices # 수신 노드 (그리드 인덱스)

    # Precompute structural node and edge features according to config options.
    # 한글 주석: (위와 동일한 설명)
    assert self._mesh_nodes_lat is not None and self._mesh_nodes_lon is not None
    (senders_node_features, receivers_node_features,
     edge_features) = model_utils.get_bipartite_graph_spatial_features(
         senders_node_lat=self._mesh_nodes_lat, # 송신 노드(메쉬) 위도
         senders_node_lon=self._mesh_nodes_lon, # 송신 노드(메쉬) 경도
         receivers_node_lat=self._grid_nodes_lat, # 수신 노드(그리드) 위도
         receivers_node_lon=self._grid_nodes_lon, # 수신 노드(그리드) 경도
         senders=senders,
         receivers=receivers,
         edge_normalization_factor=self._mesh2grid_edge_normalization_factor, # 설정된 에지 정규화 인자 사용
         **self._spatial_features_kwargs,
     )

    n_grid_node = np.array([self._num_grid_nodes])
    n_mesh_node = np.array([self._num_mesh_nodes])
    n_edge = np.array([senders.shape[0]])
    grid_node_set = typed_graph.NodeSet(
        n_node=n_grid_node, features=receivers_node_features) # 수신자(그리드)의 특징 사용
    mesh_node_set = typed_graph.NodeSet(
        n_node=n_mesh_node, features=senders_node_features) # 송신자(메쉬)의 특징 사용
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
    grid2mesh_out = self._grid2mesh_gnn(input_graph) # GNN 호출
    latent_mesh_nodes = grid2mesh_out.nodes["mesh_nodes"].features # 결과 메쉬 노드 잠재 특징
    latent_grid_nodes = grid2mesh_out.nodes["grid_nodes"].features # 결과 그리드 노드 잠재 특징
    return latent_mesh_nodes, latent_grid_nodes

  def _run_mesh_gnn(self, latent_mesh_nodes: chex.Array) -> chex.Array:
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
    return self._mesh_gnn(input_graph).nodes["mesh_nodes"].features # 결과 메쉬 노드 특징 반환

  def _run_mesh2grid_gnn(self,
                         updated_latent_mesh_nodes: chex.Array,
                         latent_grid_nodes: chex.Array,
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
    output_graph = self._mesh2grid_gnn(input_graph)
    output_grid_nodes = output_graph.nodes["grid_nodes"].features # 결과 그리드 노드 특징 (최종 출력)

    return output_grid_nodes

  def _inputs_to_grid_node_features(
      self,
      inputs: xarray.Dataset,
      forcings: xarray.Dataset,
      ) -> chex.Array:
    """xarrays -> [num_grid_nodes, batch, num_channels]."""
    # 한글 주석: xarray 데이터셋(입력, 강제 변수)을 [그리드_노드_수, 배치, 채널_수] 형태의 배열로 변환합니다.

    # xarray `Dataset` (batch, time, lat, lon, level, multiple vars)
    # to xarray `DataArray` (batch, lat, lon, channels)
    # 한글 주석: 입력과 강제 변수를 쌓아서 단일 DataArray로 만듭니다.
    stacked_inputs = model_utils.dataset_to_stacked(inputs) # 입력을 (배치, 위도, 경도, 채널) 형태로 변환
    stacked_forcings = model_utils.dataset_to_stacked(forcings) # 강제 변수를 (배치, 위도, 경도, 채널) 형태로 변환
    stacked_inputs = xarray.concat( # 입력과 강제 변수를 채널 차원으로 연결
        [stacked_inputs, stacked_forcings], dim="channels")

    # xarray `DataArray` (batch, lat, lon, channels)
    # to single numpy array with shape [lat_lon_node, batch, channels]
    # 한글 주석: (배치, 위도, 경도, 채널) 형태의 DataArray를 (위도*경도_노드, 배치, 채널) 형태의 단일 배열로 변환합니다.
    grid_xarray_lat_lon_leading = model_utils.lat_lon_to_leading_axes( # 위도, 경도 축을 맨 앞으로
        stacked_inputs)
    # 최종적으로 (노드, 배치, 특징) 형태로 변환합니다.
    return xarray_jax.unwrap(grid_xarray_lat_lon_leading.data).reshape(
        (-1,) + grid_xarray_lat_lon_leading.data.shape[2:])

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
  # 한글 주석: 메쉬 내의 최대 에지 거리를 계산합니다.
  senders, receivers = icosahedral_mesh.faces_to_edges(mesh.faces) # 면에서 에지 추출
  # 각 에지의 유클리드 거리 계산
  edge_distances = np.linalg.norm(
      mesh.vertices[senders] - mesh.vertices[receivers], axis=-1)
  return edge_distances.max() # 최대 거리 반환
