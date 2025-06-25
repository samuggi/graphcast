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
"""JAX implementation of Graph Networks Simulator.
# 한글 주석: 그래프 신경망 시뮬레이터의 JAX 구현입니다.

Generalization to TypedGraphs of the deep Graph Neural Network from:
# 한글 주석: 다음 논문들에서 제안된 심층 그래프 신경망을 TypedGraphs로 일반화한 것입니다:

@inproceedings{pfaff2021learning,
  title={Learning Mesh-Based Simulation with Graph Networks},
  author={Pfaff, Tobias and Fortunato, Meire and Sanchez-Gonzalez, Alvaro and
      Battaglia, Peter},
  booktitle={International Conference on Learning Representations},
  year={2021}
}

@inproceedings{sanchez2020learning,
  title={Learning to simulate complex physics with graph networks},
  author={Sanchez-Gonzalez, Alvaro and Godwin, Jonathan and Pfaff, Tobias and
      Ying, Rex and Leskovec, Jure and Battaglia, Peter},
  booktitle={International conference on machine learning},
  pages={8459--8468},
  year={2020},
  organization={PMLR}
}
"""

import functools
from typing import Callable, List, Mapping, Optional, Tuple # 한글 주석: 타입 힌트를 위한 모듈 임포트

import chex
from graphcast import mlp as mlp_builder # 한글 주석: MLP 빌더 모듈 임포트
from graphcast import typed_graph # 한글 주석: TypedGraph 관련 모듈 임포트
from graphcast import typed_graph_net # 한글 주석: TypedGraph 신경망 관련 모듈 임포트
import haiku as hk # Haiku: JAX를 위한 신경망 라이브러리
import jax
import jax.numpy as jnp
import jraph # Jraph: JAX를 위한 그래프 신경망 라이브러리


GraphToGraphNetwork = Callable[[typed_graph.TypedGraph], typed_graph.TypedGraph]
# 한글 주석: TypedGraph를 입력받아 TypedGraph를 출력하는 함수 타입을 정의합니다.


class DeepTypedGraphNet(hk.Module):
  """Deep Graph Neural Network.
  # 한글 주석: 심층 그래프 신경망 (Deep Graph Neural Network) 클래스입니다.

  It works with TypedGraphs with typed nodes and edges. It runs message
  passing on all of the node sets and all of the edge sets in the graph. For
  each message passing step a `typed_graph_net.InteractionNetwork` is used to
  update the full TypedGraph by using different MLPs for each of the node sets
  and each of the edge sets.
  # 한글 주석: 타입이 지정된 노드와 에지를 가진 TypedGraph에서 작동합니다. 그래프의 모든 노드 집합과 에지 집합에 대해 메시지 전달을 수행합니다.
  # 각 메시지 전달 단계에서는 `typed_graph_net.InteractionNetwork`가 사용되어 각 노드 집합과 에지 집합에 대해 서로 다른 MLP를 사용하여 전체 TypedGraph를 업데이트합니다.

  If embed_{nodes,edges} is specified the node/edge features will be embedded
  into a fixed dimensionality before running the first step of message passing.
  # 한글 주석: embed_{nodes,edges}가 지정되면, 첫 번째 메시지 전달 단계를 실행하기 전에 노드/에지 특징이 고정된 차원으로 임베딩됩니다.

  If {node,edge}_output_size the final node/edge features will be embedded into
  the specified output size.
  # 한글 주석: {node,edge}_output_size가 지정되면, 최종 노드/에지 특징이 지정된 출력 크기로 임베딩됩니다.

  This class may be used for shared or unshared message passing:
  # 한글 주석: 이 클래스는 공유 또는 비공유 메시지 전달에 사용될 수 있습니다:
  * num_message_passing_steps = N, num_processor_repetitions = 1, gives
    N layers of message passing with fully unshared weights:
    [W_1, W_2, ... , W_M] (default)
    # 한글 주석: * num_message_passing_steps = N, num_processor_repetitions = 1: 완전히 비공유된 가중치를 가진 N개의 메시지 전달 계층을 제공합니다 (기본값).
  * num_message_passing_steps = 1, num_processor_repetitions = M, gives
    N layers of message passing with fully shared weights:
    [W_1] * M
    # 한글 주석: * num_message_passing_steps = 1, num_processor_repetitions = M: 완전히 공유된 가중치를 가진 N개의 메시지 전달 계층을 제공합니다. (역자 주: 설명이 N layers로 되어 있으나 M layers가 맞을 것으로 보임)
  * num_message_passing_steps = N, num_processor_repetitions = M, gives
    M*N layers of message passing with both shared and unshared message passing
    such that the weights used at each iteration are:
    [W_1, W_2, ... , W_N] * M
    # 한글 주석: * num_message_passing_steps = N, num_processor_repetitions = M: 공유 및 비공유 메시지 전달을 모두 사용하여 M*N개의 메시지 전달 계층을 제공하며, 각 반복에서 사용되는 가중치는 [W_1, ..., W_N]이 M번 반복됩니다.
  """

  def __init__(self,
               *,
               node_latent_size: Mapping[str, int],
               edge_latent_size: Mapping[str, int],
               mlp_hidden_size: int,
               mlp_num_hidden_layers: int,
               num_message_passing_steps: int,
               num_processor_repetitions: int = 1,
               embed_nodes: bool = True,
               embed_edges: bool = True,
               node_output_size: Optional[Mapping[str, int]] = None,
               edge_output_size: Optional[Mapping[str, int]] = None,
               include_sent_messages_in_node_update: bool = False,
               use_layer_norm: bool = True,
               use_norm_conditioning: bool = False,
               activation: str = "relu",
               f32_aggregation: bool = False,
               aggregate_edges_for_nodes_fn: str = "segment_sum",
               aggregate_normalization: Optional[float] = None,
               name: str = "DeepTypedGraphNet"):
    """Inits the model.
    # 한글 주석: 모델을 초기화합니다.

    Args:
    # 한글 주석: 인수
      node_latent_size: Size of the node latent representations.
      # 한글 주석: node_latent_size: 노드 잠재 표현의 크기입니다. (노드 타입별로 지정)
      edge_latent_size: Size of the edge latent representations.
      # 한글 주석: edge_latent_size: 에지 잠재 표현의 크기입니다. (에지 타입별로 지정)
      mlp_hidden_size: Hidden layer size for all MLPs.
      # 한글 주석: mlp_hidden_size: 모든 MLP의 은닉층 크기입니다.
      mlp_num_hidden_layers: Number of hidden layers in all MLPs.
      # 한글 주석: mlp_num_hidden_layers: 모든 MLP의 은닉층 수입니다.
      num_message_passing_steps: Number of unshared message passing steps
         in the processor steps.
      # 한글 주석: num_message_passing_steps: 프로세서 단계에서 비공유 메시지 전달 단계의 수입니다.
      num_processor_repetitions: Number of times that the same processor is
         applied sequencially.
      # 한글 주석: num_processor_repetitions: 동일한 프로세서가 순차적으로 적용되는 횟수입니다.
      embed_nodes: If False, the node embedder will be omitted.
      # 한글 주석: embed_nodes: False이면 노드 임베더가 생략됩니다.
      embed_edges: If False, the edge embedder will be omitted.
      # 한글 주석: embed_edges: False이면 에지 임베더가 생략됩니다.
      node_output_size: Size of the output node representations for
         each node type. For node types not specified here, the latent node
         representation from the output of the processor will be returned.
      # 한글 주석: node_output_size: 각 노드 타입에 대한 출력 노드 표현의 크기입니다. 여기에 지정되지 않은 노드 타입의 경우, 프로세서 출력의 잠재 노드 표현이 반환됩니다.
      edge_output_size: Size of the output edge representations for
         each edge type. For edge types not specified here, the latent edge
         representation from the output of the processor will be returned.
      # 한글 주석: edge_output_size: 각 에지 타입에 대한 출력 에지 표현의 크기입니다. 여기에 지정되지 않은 에지 타입의 경우, 프로세서 출력의 잠재 에지 표현이 반환됩니다.
      include_sent_messages_in_node_update: Whether to include pooled sent
          messages from each node in the node update.
      # 한글 주석: include_sent_messages_in_node_update: 노드 업데이트 시 각 노드에서 풀링된 전송 메시지를 포함할지 여부입니다.
      use_layer_norm: Whether it uses layer norm or not.
      # 한글 주석: use_layer_norm: 레이어 정규화를 사용할지 여부입니다.
      use_norm_conditioning: If True, the latent feaures outputted by the
        activation normalization that follows the MLPs (e.g. LayerNorm), rather
        than being scaled/offset by learned  parameters of the normalization
        module, will be scaled/offset by offsets/biases produced by a linear
        layer (with different weights for each MLP), which takes an extra
        argument "global_norm_conditioning". This argument is used to condition
        the normalization of all nodes and all edges (hence global), and would
        usually only have a batch and feature axis. This is typically used to
        condition diffusion models on the "diffusion time". Will raise an error
        if this is set to True but the "global_norm_conditioning" is not passed
        to the __call__ method, as well as if this is set to False, but
        "global_norm_conditioning" is passed to the call method.
      # 한글 주석: use_norm_conditioning: True인 경우, MLP 다음에 오는 활성화 정규화(예: LayerNorm)에 의해 출력된 잠재 특징은
      # 정규화 모듈의 학습된 파라미터에 의해 스케일링/오프셋되는 대신, 추가 인수 "global_norm_conditioning"을 받는
      # 선형 계층(각 MLP에 대해 다른 가중치 사용)에 의해 생성된 오프셋/편향에 의해 스케일링/오프셋됩니다.
      # 이 인수는 모든 노드와 에지(따라서 전역적)의 정규화를 조건화하는 데 사용되며, 일반적으로 배치 및 특징 축만 갖습니다.
      # 이는 일반적으로 확산 모델을 "확산 시간"에 조건화하는 데 사용됩니다.
      # 이 값이 True이지만 "global_norm_conditioning"이 __call__ 메서드에 전달되지 않거나,
      # 이 값이 False이지만 "global_norm_conditioning"이 __call__ 메서드에 전달되면 오류가 발생합니다.
      activation: name of activation function.
      # 한글 주석: activation: 활성화 함수의 이름입니다.
      f32_aggregation: Use float32 in the edge aggregation.
      # 한글 주석: f32_aggregation: 에지 집계 시 float32를 사용합니다.
      aggregate_edges_for_nodes_fn: function used to aggregate messages to each
        node.
      # 한글 주석: aggregate_edges_for_nodes_fn: 각 노드로 메시지를 집계하는 데 사용되는 함수입니다.
      aggregate_normalization: An optional constant that normalizes the output
        of aggregate_edges_for_nodes_fn. For context, this can be used to
        reduce the shock the model undergoes when switching resolution, which
        increase the number of edges connected to a node. In particular, this is
        useful when using segment_sum, but should not be combined with
        segment_mean.
      # 한글 주석: aggregate_normalization: aggregate_edges_for_nodes_fn의 출력을 정규화하는 선택적 상수입니다.
      # 예를 들어, 해상도 변경 시 노드에 연결된 에지 수가 증가할 때 모델이 겪는 충격을 줄이는 데 사용할 수 있습니다.
      # 특히 segment_sum을 사용할 때 유용하지만 segment_mean과 함께 사용해서는 안 됩니다.
      name: Name of the model.
      # 한글 주석: name: 모델의 이름입니다.
    """

    super().__init__(name=name)

    self._node_latent_size = node_latent_size
    self._edge_latent_size = edge_latent_size
    self._mlp_hidden_size = mlp_hidden_size
    self._mlp_num_hidden_layers = mlp_num_hidden_layers
    self._num_message_passing_steps = num_message_passing_steps
    self._num_processor_repetitions = num_processor_repetitions
    self._embed_nodes = embed_nodes
    self._embed_edges = embed_edges
    self._node_output_size = node_output_size
    self._edge_output_size = edge_output_size
    self._include_sent_messages_in_node_update = (
        include_sent_messages_in_node_update)
    if use_norm_conditioning and not use_layer_norm:
      raise ValueError(
          "`norm_conditioning`은 `use_layer_norm`이 true일 때만 사용할 수 있습니다."
      )
    self._use_layer_norm = use_layer_norm
    self._use_norm_conditioning = use_norm_conditioning
    self._activation = _get_activation_fn(activation)
    self._f32_aggregation = f32_aggregation
    self._aggregate_edges_for_nodes_fn = _get_aggregate_edges_for_nodes_fn(
        aggregate_edges_for_nodes_fn)
    self._aggregate_normalization = aggregate_normalization

    if aggregate_normalization:
      # using aggregate_normalization only makes sense with segment_sum.
      # 한글 주석: aggregate_normalization은 segment_sum과 함께 사용할 때만 의미가 있습니다.
      assert aggregate_edges_for_nodes_fn == "segment_sum"

  def __call__(self,
               input_graph: typed_graph.TypedGraph,
               global_norm_conditioning: Optional[chex.Array] = None
               ) -> typed_graph.TypedGraph:
    """Forward pass of the learnable dynamics model."""
    # 한글 주석: 학습 가능한 동역학 모델의 순전파를 수행합니다.
    embedder_network, processor_networks, decoder_network = (
        self._networks_builder(input_graph, global_norm_conditioning)
    )
    # 한글 주석: 인코더(임베더), 프로세서, 디코더 네트워크를 구성합니다.

    # Embed input features (if applicable).
    # 한글 주석: 입력 특징을 임베딩합니다 (해당하는 경우).
    latent_graph_0 = self._embed(input_graph, embedder_network)

    # Do `m` message passing steps in the latent graphs.
    # 한글 주석: 잠재 그래프에서 `m`개의 메시지 전달 단계를 수행합니다.
    latent_graph_m = self._process(latent_graph_0, processor_networks)

    # Compute outputs from the last latent graph (if applicable).
    # 한글 주석: 마지막 잠재 그래프에서 출력을 계산합니다 (해당하는 경우).
    return self._output(latent_graph_m, decoder_network)

  def _networks_builder(
      self,
      graph_template: typed_graph.TypedGraph,
      global_norm_conditioning: Optional[chex.Array] = None,
  ) -> Tuple[
      GraphToGraphNetwork, List[GraphToGraphNetwork], GraphToGraphNetwork
  ]:
    """네트워크 빌더: 인코더, 프로세서, 디코더 네트워크를 생성합니다."""
    # 한글 주석: 이 메서드는 모델의 주요 구성 요소인 인코더, 여러 개의 프로세서, 디코더 네트워크를 생성합니다.
    # TODO(aelkadi): move to mlp_builder. # 한글 주석: TODO: mlp_builder로 옮길 것.
    def build_mlp(name, output_size):
      # 한글 주석: 주어진 이름과 출력 크기로 MLP를 생성하는 헬퍼 함수입니다.
      mlp = hk.nets.MLP(
          output_sizes=[self._mlp_hidden_size] * self._mlp_num_hidden_layers + [
              output_size], name=name + "_mlp", activation=self._activation)
      return jraph.concatenated_args(mlp) # 입력 인자들을 연결하여 MLP에 전달합니다.

    def build_mlp_with_maybe_layer_norm(name, output_size):
      # 한글 주석: MLP와 선택적으로 레이어 정규화를 포함하는 네트워크를 생성하는 헬퍼 함수입니다.
      network = build_mlp(name, output_size) # 기본 MLP 생성
      stages = [network] # 네트워크 단계를 리스트로 관리

      if self._use_norm_conditioning: # 정규화 조건화 사용 시
        if global_norm_conditioning is None:
          raise ValueError(
              "정규화 조건화 사용 시 `global_norm_conditioning`이 __call__ 메서드에 전달되어야 합니다.")
        # If using norm conditioning, it is no longer the responsibility of the
        # LayerNorm module itself to learn its scale and offset. These will be
        # learned for the module by the norm conditioning layer instead.
        # 한글 주석: 정규화 조건화를 사용하는 경우, LayerNorm 모듈 자체가 스케일과 오프셋을 학습하는 것이 아니라,
        # 정규화 조건화 계층이 대신 학습합니다.
        create_scale = create_offset = False # LayerNorm의 학습 가능한 스케일/오프셋 비활성화
      else: # 정규화 조건화 미사용 시
        if global_norm_conditioning is not None:
          raise ValueError(
              "`global_norm_conditioning`이 전달되었지만 `norm_conditioning`이 활성화되지 않았습니다.")
        create_scale = create_offset = True # LayerNorm의 학습 가능한 스케일/오프셋 활성화

      if self._use_layer_norm: # 레이어 정규화 사용 시
        layer_norm = hk.LayerNorm(
            axis=-1, create_scale=create_scale, create_offset=create_offset,
            name=name + "_layer_norm")
        stages.append(layer_norm) # 레이어 정규화 단계를 추가

      if self._use_norm_conditioning: # 정규화 조건화 사용 시
        norm_conditioning_layer = mlp_builder.LinearNormConditioning(
            name=name + "_norm_conditioning")
        norm_conditioning_layer = functools.partial( # 부분 적용 함수 생성
            norm_conditioning_layer,
            # Broadcast to the node/edge axis.
            # 한글 주석: 노드/에지 축으로 브로드캐스팅합니다.
            norm_conditioning=global_norm_conditioning[None], # 전역 정규화 조건 전달
        )
        stages.append(norm_conditioning_layer) # 정규화 조건화 단계를 추가

      network = hk.Sequential(stages) # 모든 단계를 순차적으로 실행하는 네트워크 생성
      return jraph.concatenated_args(network)

    # The embedder graph network independently embeds edge and node features.
    # 한글 주석: 임베더 그래프 네트워크는 에지와 노드 특징을 독립적으로 임베딩합니다.
    if self._embed_edges: # 에지 임베딩 사용 시
      embed_edge_fn = _build_update_fns_for_edge_types(
          build_mlp_with_maybe_layer_norm, # MLP 빌더 함수 (레이어 정규화 포함 가능)
          graph_template, # 그래프 템플릿 (타입 정보 포함)
          "encoder_edges_", # 이름 접두사
          output_sizes=self._edge_latent_size) # 에지 타입별 잠재 크기
    else:
      embed_edge_fn = None
    if self._embed_nodes: # 노드 임베딩 사용 시
      embed_node_fn = _build_update_fns_for_node_types(
          build_mlp_with_maybe_layer_norm,
          graph_template,
          "encoder_nodes_",
          output_sizes=self._node_latent_size) # 노드 타입별 잠재 크기
    else:
      embed_node_fn = None
    embedder_kwargs = dict( # 임베더 네트워크 생성 인자
        embed_edge_fn=embed_edge_fn,
        embed_node_fn=embed_node_fn,
    )
    embedder_network = typed_graph_net.GraphMapFeatures(**embedder_kwargs) # 임베더 네트워크 생성

    if self._f32_aggregation: # float32 집계 사용 시
      def aggregate_fn(data, *args, **kwargs):
        # 한글 주석: 에지 특징을 float32로 변환하여 집계하고, 원래 타입으로 다시 변환하는 함수입니다.
        dtype = data.dtype # 원래 데이터 타입 저장
        data = data.astype(jnp.float32) # float32로 변환
        output = self._aggregate_edges_for_nodes_fn(data, *args, **kwargs) # 지정된 집계 함수 사용
        if self._aggregate_normalization: # 집계 정규화 사용 시
          output = output / self._aggregate_normalization # 정규화 상수 적용
        output = output.astype(dtype) # 원래 데이터 타입으로 복원
        return output
    else: # float32 집계 미사용 시
      def aggregate_fn(data, *args, **kwargs):
        # 한글 주석: 지정된 집계 함수를 사용하고, 선택적으로 집계 정규화를 적용하는 함수입니다.
        output = self._aggregate_edges_for_nodes_fn(data, *args, **kwargs)
        if self._aggregate_normalization:
          output = output / self._aggregate_normalization
        return output

    # Create `num_message_passing_steps` graph networks with unshared parameters
    # that update the node and edge latent features.
    # Note that we can use `modules.InteractionNetwork` because
    # it also outputs the messages as updated edge latent features.
    # 한글 주석: 노드 및 에지 잠재 특징을 업데이트하는 비공유 파라미터를 가진 `num_message_passing_steps`개의 그래프 네트워크를 생성합니다.
    # `modules.InteractionNetwork`는 메시지를 업데이트된 에지 잠재 특징으로 출력하므로 사용할 수 있습니다.
    processor_networks = []
    for step_i in range(self._num_message_passing_steps): # 각 메시지 전달 단계에 대해
      processor_networks.append( # 프로세서 네트워크 리스트에 추가
          typed_graph_net.InteractionNetwork( # InteractionNetwork 생성
              update_edge_fn=_build_update_fns_for_edge_types( # 에지 업데이트 함수
                  build_mlp_with_maybe_layer_norm,
                  graph_template,
                  f"processor_edges_{step_i}_", # 각 단계별 고유 이름
                  output_sizes=self._edge_latent_size),
              update_node_fn=_build_update_fns_for_node_types( # 노드 업데이트 함수
                  build_mlp_with_maybe_layer_norm,
                  graph_template,
                  f"processor_nodes_{step_i}_", # 각 단계별 고유 이름
                  output_sizes=self._node_latent_size),
              aggregate_edges_for_nodes_fn=aggregate_fn, # 위에서 정의한 집계 함수
              include_sent_messages_in_node_update=( # 노드 업데이트 시 보낸 메시지 포함 여부
                  self._include_sent_messages_in_node_update),
              ))

    # The output MLPs converts edge/node latent features into the output sizes.
    # 한글 주석: 출력 MLP는 에지/노드 잠재 특징을 지정된 출력 크기로 변환합니다.
    output_kwargs = dict( # 디코더 네트워크 생성 인자
        embed_edge_fn=_build_update_fns_for_edge_types( # 에지 출력 함수
            build_mlp, graph_template, "decoder_edges_", self._edge_output_size)
        if self._edge_output_size else None, # 에지 출력 크기가 지정된 경우에만 생성
        embed_node_fn=_build_update_fns_for_node_types( # 노드 출력 함수
            build_mlp, graph_template, "decoder_nodes_", self._node_output_size)
        if self._node_output_size else None, # 노드 출력 크기가 지정된 경우에만 생성
    )
    output_network = typed_graph_net.GraphMapFeatures(**output_kwargs) # 디코더 네트워크 생성
    return embedder_network, processor_networks, output_network

  def _embed(
      self,
      input_graph: typed_graph.TypedGraph,
      embedder_network: GraphToGraphNetwork,
  ) -> typed_graph.TypedGraph:
    """Embeds the input graph features into a latent graph."""
    # 한글 주석: 입력 그래프 특징을 잠재 그래프로 임베딩합니다.

    # Copy the context to all of the node types, if applicable.
    # 한글 주석: 해당되는 경우, 컨텍스트를 모든 노드 타입에 복사합니다.
    context_features = input_graph.context.features # 그래프 컨텍스트 특징
    if jax.tree_util.tree_leaves(context_features): # 컨텍스트 특징이 비어있지 않은 경우
      # This code assumes a single input feature array for the context and for
      # each node type.
      # 한글 주석: 이 코드는 컨텍스트와 각 노드 타입에 대해 단일 입력 특징 배열을 가정합니다.
      assert len(jax.tree_util.tree_leaves(context_features)) == 1
      new_nodes = {}
      for node_set_name, node_set in input_graph.nodes.items(): # 각 노드 집합에 대해
        node_features = node_set.features # 현재 노드 특징
        # 컨텍스트 특징을 노드 수만큼 반복하여 노드 특징과 연결할 수 있도록 브로드캐스팅합니다.
        broadcasted_context = jnp.repeat(
            context_features, node_set.n_node, axis=0,
            total_repeat_length=node_features.shape[0])
        new_nodes[node_set_name] = node_set._replace( # 새로운 노드 특징으로 업데이트
            features=jnp.concatenate( # 기존 노드 특징과 브로드캐스팅된 컨텍스트 특징을 연결
                [node_features, broadcasted_context], axis=-1))
      input_graph = input_graph._replace( # 업데이트된 노드들로 그래프 교체
          nodes=new_nodes,
          context=input_graph.context._replace(features=())) # 컨텍스트 특징은 비움 (이미 노드에 전달됨)

    # Embeds the node and edge features.
    # 한글 주석: 노드와 에지 특징을 임베딩합니다.
    latent_graph_0 = embedder_network(input_graph) # 임베더 네트워크 통과
    return latent_graph_0

  def _process(
      self,
      latent_graph_0: typed_graph.TypedGraph,
      processor_networks: List[GraphToGraphNetwork],
  ) -> typed_graph.TypedGraph:
    """Processes the latent graph with several steps of message passing."""
    # 한글 주석: 여러 단계의 메시지 전달을 통해 잠재 그래프를 처리합니다.

    # Do `num_message_passing_steps` with each of the `self._processor_networks`
    # with unshared weights, and repeat that `self._num_processor_repetitions`
    # times.
    # 한글 주석: 비공유 가중치를 가진 `self._processor_networks` 각각에 대해 `num_message_passing_steps`를 수행하고,
    # 이를 `self._num_processor_repetitions`번 반복합니다.
    latent_graph = latent_graph_0 # 초기 잠재 그래프
    for unused_repetition_i in range(self._num_processor_repetitions): # 프로세서 반복 횟수만큼
      for processor_network in processor_networks: # 각 프로세서 네트워크에 대해 (비공유 가중치)
        latent_graph = self._process_step(processor_network, latent_graph) # 단일 처리 단계 수행

    return latent_graph

  def _process_step(
      self, processor_network_k,
      latent_graph_prev_k: typed_graph.TypedGraph) -> typed_graph.TypedGraph:
    """Single step of message passing with node/edge residual connections."""
    # 한글 주석: 노드/에지 잔차 연결을 포함하는 단일 메시지 전달 단계입니다.

    # One step of message passing.
    # 한글 주석: 한 단계의 메시지 전달을 수행합니다.
    latent_graph_k = processor_network_k(latent_graph_prev_k) # 현재 프로세서 네트워크 통과

    # Add residuals.
    # 한글 주석: 잔차 연결을 추가합니다.
    nodes_with_residuals = {}
    for k, prev_set in latent_graph_prev_k.nodes.items(): # 이전 잠재 그래프의 각 노드 집합에 대해
      nodes_with_residuals[k] = prev_set._replace( # 잔차 연결 추가 (이전 특징 + 현재 특징)
          features=prev_set.features + latent_graph_k.nodes[k].features)

    edges_with_residuals = {}
    for k, prev_set in latent_graph_prev_k.edges.items(): # 이전 잠재 그래프의 각 에지 집합에 대해
      edges_with_residuals[k] = prev_set._replace( # 잔차 연결 추가
          features=prev_set.features + latent_graph_k.edges[k].features)

    latent_graph_k = latent_graph_k._replace( # 잔차 연결이 추가된 노드와 에지로 그래프 업데이트
        nodes=nodes_with_residuals, edges=edges_with_residuals)
    return latent_graph_k

  def _output(
      self,
      latent_graph: typed_graph.TypedGraph,
      output_network: GraphToGraphNetwork,
  ) -> typed_graph.TypedGraph:
    """Produces the output from the latent graph."""
    # 한글 주석: 잠재 그래프로부터 최종 출력을 생성합니다.
    return output_network(latent_graph) # 디코더(출력) 네트워크 통과


def _build_update_fns_for_node_types(
    builder_fn, graph_template, prefix, output_sizes=None):
  """Builds an update function for all node types or a subset of them."""
  # 한글 주석: 모든 노드 타입 또는 일부 노드 타입에 대한 업데이트 함수를 빌드합니다.

  output_fns = {} # 노드 타입별 업데이트 함수를 저장할 딕셔너리
  for node_set_name in graph_template.nodes.keys(): # 그래프 템플릿의 모든 노드 타입 이름에 대해
    if output_sizes is None: # 출력 크기가 지정되지 않은 경우
      # Use the default output size for all types.
      # 한글 주석: 모든 타입에 대해 기본 출력 크기를 사용합니다 (builder_fn 내부에서 결정될 수 있음).
      output_size = None
    else: # 출력 크기가 지정된 경우
      # Otherwise, ignore any type that does not have an explicit output size.
      # 한글 주석: 명시적인 출력 크기가 없는 타입은 무시합니다.
      if node_set_name in output_sizes:
        output_size = output_sizes[node_set_name] # 해당 노드 타입의 출력 크기 사용
      else:
        continue # 해당 노드 타입에 대한 업데이트 함수는 생성하지 않음
    output_fns[node_set_name] = builder_fn( # 빌더 함수를 사용하여 업데이트 함수 생성
        f"{prefix}{node_set_name}", output_size) # 이름과 출력 크기 전달
  return output_fns


def _build_update_fns_for_edge_types(
    builder_fn, graph_template, prefix, output_sizes=None):
  """Builds an edge function for all node types or a subset of them."""
  # 한글 주석: 모든 에지 타입 또는 일부 에지 타입에 대한 업데이트 함수를 빌드합니다.
  # (주석 오타 수정: node types -> edge types)
  output_fns = {} # 에지 타입별 업데이트 함수를 저장할 딕셔너리
  for edge_set_key in graph_template.edges.keys(): # 그래프 템플릿의 모든 에지 타입 키에 대해
    edge_set_name = edge_set_key.name # 에지 타입 이름
    if output_sizes is None: # 출력 크기가 지정되지 않은 경우
      # Use the default output size for all types.
      # 한글 주석: 모든 타입에 대해 기본 출력 크기를 사용합니다.
      output_size = None
    else: # 출력 크기가 지정된 경우
      # Otherwise, ignore any type that does not have an explicit output size.
      # 한글 주석: 명시적인 출력 크기가 없는 타입은 무시합니다.
      if edge_set_name in output_sizes:
        output_size = output_sizes[edge_set_name] # 해당 에지 타입의 출력 크기 사용
      else:
        continue # 해당 에지 타입에 대한 업데이트 함수는 생성하지 않음
    output_fns[edge_set_name] = builder_fn( # 빌더 함수를 사용하여 업데이트 함수 생성
        f"{prefix}{edge_set_name}", output_size) # 이름과 출력 크기 전달
  return output_fns


def _get_activation_fn(name):
  """Return activation function corresponding to function_name."""
  # 한글 주석: 함수 이름에 해당하는 활성화 함수를 반환합니다.
  if name == "identity": # "identity"인 경우 항등 함수 반환
    return lambda x: x
  if hasattr(jax.nn, name): # jax.nn 모듈에 해당 이름의 함수가 있으면 반환
    return getattr(jax.nn, name)
  if hasattr(jnp, name): # jax.numpy 모듈에 해당 이름의 함수가 있으면 반환
    return getattr(jnp, name)
  raise ValueError(f"알 수 없는 활성화 함수 {name}이(가) 지정되었습니다.")


def _get_aggregate_edges_for_nodes_fn(name):
  """Return aggregate_edges_for_nodes_fn corresponding to function_name."""
  # 한글 주석: 함수 이름에 해당하는 aggregate_edges_for_nodes_fn (에지 집계 함수)을 반환합니다.
  if hasattr(jraph, name): # jraph 모듈에 해당 이름의 함수가 있으면 반환
    return getattr(jraph, name)
  raise ValueError(
      f"알 수 없는 aggregate_edges_for_nodes_fn 함수 {name}이(가) 지정되었습니다.")
