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
"""한 단계 예측기를 래핑하여 자동 회귀 예측을 수행하는 예측기입니다.
"""
# 한글 주석: 위 설명은 이 파일이 단일 단계 예측기를 사용하여 여러 단계에 걸쳐 자동 회귀적으로 예측을 수행하는 기능을 제공한다는 의미입니다.

from typing import Optional, cast

from absl import logging
from graphcast import predictor_base
from graphcast import xarray_jax
from graphcast import xarray_tree
import haiku as hk
import jax
import xarray


def _unflatten_and_expand_time(flat_variables, tree_def, time_coords):
  """평탄화된 변수를 원래 트리 구조로 복원하고 시간 차원을 확장합니다."""
  # 한글 주석: 이 함수는 JAX 트리 유틸리티를 사용하여 평탄화된 변수 목록을 원래의 중첩된 데이터 구조(트리)로 되돌립니다.
  # 그런 다음 xarray를 사용하여 지정된 시간 좌표로 시간 차원을 추가하고 확장합니다.
  variables = jax.tree_util.tree_unflatten(tree_def, flat_variables)
  return variables.expand_dims(time=time_coords, axis=0)


def _get_flat_arrays_and_single_timestep_treedef(variables):
  """변수에서 평탄화된 배열과 단일 시간 단계 트리 정의를 가져옵니다."""
  # 한글 주석: 이 함수는 xarray 데이터셋을 JAX 트리 유틸리티가 처리할 수 있는 평탄화된 배열 목록으로 변환합니다.
  # 또한, 단일 시간 단계에 대한 트리 구조 정의(treedef)도 반환하여 나중에 데이터를 재구성하는 데 사용됩니다.
  # 'time' 차원을 첫 번째로 옮긴 후 평탄화를 수행합니다.
  flat_arrays = jax.tree_util.tree_leaves(variables.transpose('time', ...))
  # 첫 번째 시간 단계의 데이터만 선택하고(drop=True로 차원 유지 안 함) 트리 구조를 얻습니다.
  _, treedef = jax.tree_util.tree_flatten(variables.isel(time=0, drop=True))
  return flat_arrays, treedef


class Predictor(predictor_base.Predictor):
  """Wraps a one-step Predictor to make multi-step predictions autoregressively.
  # 한글 주석: 단일 단계 예측기를 래핑하여 여러 단계의 예측을 자동 회귀적으로 수행하는 클래스입니다.

  The wrapped Predictor will be used to predict a single timestep conditional
  on the inputs passed to the outer Predictor. Its predictions are then
  passed back in as inputs at the next timestep, for as many timesteps as are
  requested in the targets_template. (When multiple timesteps of input are
  used, a rolling window of inputs is maintained with new predictions
  concatenated onto the end).
  # 한글 주석: 래핑된 예측기는 외부 예측기에 전달된 입력을 조건으로 단일 시간 단계를 예측하는 데 사용됩니다.
  # 그런 다음 예측 결과는 targets_template에 요청된 시간 단계만큼 다음 시간 단계의 입력으로 다시 전달됩니다.
  # (여러 시간 단계의 입력이 사용되는 경우, 새로운 예측이 끝에 연결된 입력의 롤링 윈도우가 유지됩니다.)

  You may ask for additional variables to be predicted as targets which aren't
  used as inputs. These will be predicted as output variables only and not fed
  back in autoregressively. All target variables must be time-dependent however.
  # 한글 주석: 입력으로 사용되지 않는 추가 변수를 대상으로 예측하도록 요청할 수 있습니다.
  # 이러한 변수는 출력 변수로만 예측되며 자동 회귀적으로 다시 입력되지 않습니다.
  # 그러나 모든 대상 변수는 시간에 따라 달라야 합니다.

  You may also specify static (non-time-dependent) inputs which will be passed
  in at each timestep but are not predicted.
  # 한글 주석: 각 시간 단계에 전달되지만 예측되지는 않는 정적(시간에 의존하지 않는) 입력을 지정할 수도 있습니다.

  At present, any time-dependent inputs must also be present as targets so they
  can be passed in autoregressively.
  # 한글 주석: 현재, 시간에 따라 변하는 모든 입력은 자동 회귀적으로 전달될 수 있도록 대상에도 존재해야 합니다.

  The loss of the wrapped one-step Predictor is averaged over all timesteps to
  give a loss for the autoregressive Predictor.
  # 한글 주석: 래핑된 단일 단계 예측기의 손실은 모든 시간 단계에 대해 평균화되어 자동 회귀 예측기의 손실을 제공합니다.
  """

  def __init__(
      self,
      predictor: predictor_base.Predictor,
      noise_level: Optional[float] = None,
      gradient_checkpointing: bool = False,
      ):
    """Initializes an autoregressive predictor wrapper.
    # 한글 주석: 자동 회귀 예측기 래퍼를 초기화합니다.

    Args:
    # 한글 주석: 인수
      predictor: A predictor to wrap in an auto-regressive way.
      # 한글 주석: predictor: 자동 회귀 방식으로 래핑할 예측기입니다.
      noise_level: Optional value that multiplies the standard normal noise
        added to the time-dependent variables of the predictor inputs. In
        particular, no noise is added to the predictions that are fed back
        auto-regressively. Defaults to not adding noise.
      # 한글 주석: noise_level: 예측기 입력의 시간에 따라 변하는 변수에 추가되는 표준 정규 노이즈에 곱하는 선택적 값입니다.
      # 특히, 자동 회귀적으로 다시 입력되는 예측에는 노이즈가 추가되지 않습니다. 기본값은 노이즈를 추가하지 않는 것입니다.
      gradient_checkpointing: If True, gradient checkpointing will be
        used at each step of the computation to save on memory. Roughtly this
        should make the backwards pass two times more expensive, and the time
        per step counting the forward pass, should only increase by about 50%.
        Note this parameter will be ignored with a warning if the scan sequence
        length is 1.
      # 한글 주석: gradient_checkpointing: True이면 메모리 절약을 위해 계산의 각 단계에서 그래디언트 체크포인팅이 사용됩니다.
      # 대략적으로 역전파는 두 배 더 비싸지고, 순전파를 포함한 단계별 시간은 약 50%만 증가해야 합니다.
      # 스캔 시퀀스 길이가 1이면 이 매개변수는 경고와 함께 무시됩니다.
    """
    self._predictor = predictor
    self._noise_level = noise_level
    self._gradient_checkpointing = gradient_checkpointing

  def _get_and_validate_constant_inputs(self, inputs, targets, forcings):
    """입력, 대상, 강제 변수에서 정적 입력을 가져오고 유효성을 검사합니다."""
    # 한글 주석: 이 메서드는 대상(targets)과 강제 변수(forcings)에 속하지 않는 입력 변수를 정적 입력으로 간주합니다.
    # 그런 다음 정적 입력이 시간 차원('time')을 가지고 있는지 확인하고, 있다면 오류를 발생시킵니다.
    # 시간 의존적 입력은 강제 변수이거나 자동 회귀 피드백을 위한 대상 변수여야 하기 때문입니다.
    constant_inputs = inputs.drop_vars(targets.keys(), errors='ignore')
    constant_inputs = constant_inputs.drop_vars(
        forcings.keys(), errors='ignore')
    for name, var in constant_inputs.items():
      if 'time' in var.dims:
        raise ValueError(
            f'시간 의존적 입력 변수 {name}은(는) 강제 변수이거나 자동 회귀 피드백을 허용하는 대상 변수여야 합니다.')
    return constant_inputs

  def _validate_targets_and_forcings(self, targets, forcings):
    """대상 및 강제 변수의 유효성을 검사합니다."""
    # 한글 주석: 이 메서드는 모든 대상 변수와 강제 변수가 시간 차원('time')을 가지고 있는지 확인합니다.
    # 또한, 대상 변수와 강제 변수 간에 겹치는 변수가 있는지 확인하고, 있다면 오류를 발생시킵니다.
    for name, var in targets.items():
      if 'time' not in var.dims:
        raise ValueError(f'대상 변수 {name}은(는) 시간에 따라 변해야 합니다.')

    for name, var in forcings.items():
      if 'time' not in var.dims:
        raise ValueError(f'강제 변수 {name}은(는) 시간에 따라 변해야 합니다.')

    overlap = forcings.keys() & targets.keys()
    if overlap:
      raise ValueError('다음 변수들은 대상과 강제 변수 모두로 지정되었으며, 이는 허용되지 않습니다: '
                       f'{overlap}')

  def _update_inputs(self, inputs, next_frame):
    """다음 시간 단계의 예측 또는 강제 변수를 사용하여 입력을 업데이트합니다."""
    # 한글 주석: 이 메서드는 현재 입력과 다음 시간 단계의 예측/강제 프레임을 결합하여 다음 예측을 위한 입력을 준비합니다.
    # 입력의 시간 차원 수를 유지하면서 가장 최근의 데이터만 사용합니다 (롤링 윈도우).
    # 또한 다음 자동 회귀 반복을 위해 시간 좌표를 재설정합니다.
    num_inputs = inputs.dims['time']

    # 입력에 존재하는 키만 사용하여 next_frame에서 변수를 선택합니다.
    predicted_or_forced_inputs = next_frame[list(inputs.keys())]

    # 입력과 대상 시간 스탬프가 있는 데이터셋을 결합하면 정렬됩니다.
    # 다음 입력으로 사용할 후행 num_inputs 프레임만 유지합니다.
    return (xarray.concat([inputs, predicted_or_forced_inputs], dim='time')
            .tail(time=num_inputs)
            # 다음 AR 반복을 위해 리드 타임을 재설정하도록 시간 좌표를 업데이트합니다.
            .assign_coords(time=inputs.coords['time']))

  def __call__(self,
               inputs: xarray.Dataset,
               targets_template: xarray.Dataset,
               forcings: xarray.Dataset,
               **kwargs) -> xarray.Dataset:
    """Calls the Predictor.
    # 한글 주석: 예측기를 호출합니다.

    Args:
    # 한글 주석: 인수
      inputs: input variable used to make predictions. Inputs can include both
        time-dependent and time independent variables. Any time-dependent
        input variables must also be present in the targets_template or the
        forcings.
      # 한글 주석: inputs: 예측을 만드는 데 사용되는 입력 변수입니다. 입력에는 시간에 따라 변하는 변수와 시간에 독립적인 변수가 모두 포함될 수 있습니다.
      # 시간에 따라 변하는 모든 입력 변수는 targets_template 또는 forcings에도 있어야 합니다.
      targets_template: A target template containing informations about which
        variables should be predicted and the time alignment of the predictions.
        All target variables must be time-dependent.
        The number of time frames is used to set the number of unroll of the AR
        predictor (e.g. multiple unroll of the inner predictor for one time step
        in the targets is not supported yet).
      # 한글 주석: targets_template: 예측할 변수와 예측의 시간 정렬에 대한 정보가 포함된 대상 템플릿입니다.
      # 모든 대상 변수는 시간에 따라 변해야 합니다.
      # 시간 프레임 수는 AR 예측기의 전개 횟수를 설정하는 데 사용됩니다 (예: 대상의 한 시간 단계에 대한 내부 예측기의 다중 전개는 아직 지원되지 않음).
      forcings: Variables that will be fed to the model. The variables
        should not overlap with the target ones. The time coordinates of the
        forcing variables should match the target ones.
        Forcing variables which are also present in the inputs, will be used to
        supply ground-truth values for those inputs when they are passed to the
        underlying predictor at timesteps beyond the first timestep.
      # 한글 주석: forcings: 모델에 입력될 변수입니다. 변수는 대상 변수와 겹치지 않아야 합니다.
      # 강제 변수의 시간 좌표는 대상 변수의 시간 좌표와 일치해야 합니다.
      # 입력에도 있는 강제 변수는 첫 번째 시간 단계 이후의 시간 단계에서 기본 예측기에 전달될 때 해당 입력에 대한 실제 값을 제공하는 데 사용됩니다.
      **kwargs: Additional arguments passed along to the inner Predictor.
      # 한글 주석: **kwargs: 내부 예측기로 전달되는 추가 인수입니다.

    Returns:
    # 한글 주석: 반환값
      predictions: the model predictions matching the target template.
      # 한글 주석: predictions: 대상 템플릿과 일치하는 모델 예측입니다.

    Raise:
    # 한글 주석: 발생 오류
      ValueError: if the time coordinates of the inputs and targets are not
        different by a constant time step.
      # 한글 주석: ValueError: 입력과 대상의 시간 좌표가 일정한 시간 단계만큼 다르지 않은 경우 발생합니다.
    """

    constant_inputs = self._get_and_validate_constant_inputs(
        inputs, targets_template, forcings)
    self._validate_targets_and_forcings(targets_template, forcings)

    # After the above checks, the remaining inputs must be time-dependent:
    # 한글 주석: 위의 확인 후, 나머지 입력은 시간에 따라 변해야 합니다.
    inputs = inputs.drop_vars(constant_inputs.keys())

    # A predictions template only including the next time to predict.
    # 한글 주석: 예측할 다음 시간만 포함하는 예측 템플릿입니다.
    target_template = targets_template.isel(time=[0])

    flat_forcings, forcings_treedef = (
        _get_flat_arrays_and_single_timestep_treedef(forcings))
    scan_variables = flat_forcings
    # 한글 주석: 강제 변수를 평탄화하고 hk.scan에 전달할 스캔 변수로 설정합니다.

    def one_step_prediction(inputs, scan_variables):
      """단일 시간 단계 예측을 수행하는 내부 함수입니다."""
      # 한글 주석: 이 함수는 hk.scan에 의해 반복적으로 호출되어 각 시간 단계의 예측을 생성합니다.

      flat_forcings = scan_variables
      forcings = _unflatten_and_expand_time(flat_forcings, forcings_treedef,
                                            target_template.coords['time'])
      # 한글 주석: 스캔 변수에서 평탄화된 강제 변수를 가져와 시간 차원을 복원합니다.

      # Add constant inputs:
      # 한글 주석: 정적 입력을 추가합니다.
      all_inputs = xarray.merge([constant_inputs, inputs])
      predictions: xarray.Dataset = self._predictor(
          all_inputs, target_template,
          forcings=forcings,
          **kwargs)
      # 한글 주석: 내부 예측기를 호출하여 예측을 생성합니다.

      next_frame = xarray.merge([predictions, forcings])
      # 한글 주석: 예측과 강제 변수를 병합하여 다음 프레임을 만듭니다.
      next_inputs = self._update_inputs(inputs, next_frame)
      # 한글 주석: 다음 반복을 위한 입력을 업데이트합니다.

      # Drop the length-1 time dimension, since scan will concat all the outputs
      # for different times along a new leading time dimension:
      # 한글 주석: 길이가 1인 시간 차원을 제거합니다. scan이 다른 시간에 대한 모든 출력을 새로운 선행 시간 차원을 따라 연결하기 때문입니다.
      predictions = predictions.squeeze('time', drop=True)
      # We return the prediction flattened into plain jax arrays, because the
      # extra leading dimension added by scan prevents the tree_util
      # registrations in xarray_jax from unflattening them back into an
      # xarray.Dataset automatically:
      # 한글 주석: 예측을 평탄화된 jax 배열로 반환합니다. scan에 의해 추가된 여분의 선행 차원으로 인해 xarray_jax의 tree_util 등록이
      # 자동으로 xarray.Dataset으로 다시 평탄화하는 것을 방지하기 때문입니다.
      flat_pred = jax.tree_util.tree_leaves(predictions)
      return next_inputs, flat_pred

    if self._gradient_checkpointing:
      # 한글 주석: 그래디언트 체크포인팅이 활성화된 경우
      scan_length = targets_template.dims['time']
      if scan_length <= 1:
        logging.warning(
            '길이가 1인 시퀀스에 대해서는 그래디언트 체크포인팅을 건너<0xEB><0xA9><0x95>니다.')
      else:
        # Just in case we take gradients (e.g. for control), although
        # in most cases this will just be for a forward pass.
        # 한글 주석: 대부분의 경우 순전파를 위한 것이지만, 제어 등을 위해 그래디언트를 사용하는 경우를 대비합니다.
        one_step_prediction = hk.remat(one_step_prediction)
        # 한글 주석: hk.remat을 사용하여 그래디언트 체크포인팅을 적용합니다.

    # Loop (without unroll) with hk states in cell (jax.lax.scan won't do).
    # 한글 주석: 셀에 hk 상태를 가진 루프 (전개 없음) (jax.lax.scan은 수행하지 않음).
    # hk.scan을 사용하여 여러 시간 단계에 걸쳐 one_step_prediction을 반복 실행합니다.
    _, flat_preds = hk.scan(one_step_prediction, inputs, scan_variables)

    # The result of scan will have an extra leading axis on all arrays,
    # corresponding to the target times in this case. We need to be prepared for
    # it when unflattening the arrays back into a Dataset:
    # 한글 주석: scan의 결과는 모든 배열에 여분의 선행 축을 갖게 되며, 이 경우 대상 시간에 해당합니다.
    # 배열을 Dataset으로 다시 평탄화할 때 이를 준비해야 합니다.
    scan_result_template = (
        target_template.squeeze('time', drop=True)
        .expand_dims(time=targets_template.coords['time'], axis=0))
    _, scan_result_treedef = jax.tree_util.tree_flatten(scan_result_template)
    predictions = jax.tree_util.tree_unflatten(scan_result_treedef, flat_preds)
    # 한글 주석: 평탄화된 예측 결과를 원래의 xarray.Dataset 구조로 복원합니다.
    return predictions

  def loss(self,
           inputs: xarray.Dataset,
           targets: xarray.Dataset,
           forcings: xarray.Dataset,
           **kwargs
           ) -> predictor_base.LossAndDiagnostics:
    """The mean of the per-timestep losses of the underlying predictor."""
    # 한글 주석: 기본 예측기의 시간 단계별 손실의 평균입니다.
    if targets.sizes['time'] == 1:
      # If there is only a single target timestep then we don't need any
      # autoregressive feedback and can delegate the loss directly to the
      # underlying single-step predictor. This means the underlying predictor
      # doesn't need to implement .loss_and_predictions.
      # 한글 주석: 대상 시간 단계가 하나만 있는 경우 자동 회귀 피드백이 필요하지 않으며
      # 손실을 기본 단일 단계 예측기에 직접 위임할 수 있습니다.
      # 이는 기본 예측기가 .loss_and_predictions를 구현할 필요가 없음을 의미합니다.
      return self._predictor.loss(inputs, targets, forcings, **kwargs)

    constant_inputs = self._get_and_validate_constant_inputs(
        inputs, targets, forcings)
    self._validate_targets_and_forcings(targets, forcings)
    # After the above checks, the remaining inputs must be time-dependent:
    # 한글 주석: 위의 확인 후, 나머지 입력은 시간에 따라 변해야 합니다.
    inputs = inputs.drop_vars(constant_inputs.keys())

    if self._noise_level:
      # 한글 주석: 노이즈 레벨이 설정된 경우
      def add_noise(x):
        # 한글 주석: 입력에 표준 정규 분포 노이즈를 추가하는 함수입니다.
        return x + self._noise_level * jax.random.normal(
            hk.next_rng_key(), shape=x.shape)
      # Add noise to time-dependent variables of the inputs.
      # 한글 주석: 입력의 시간에 따라 변하는 변수에 노이즈를 추가합니다.
      inputs = jax.tree.map(add_noise, inputs)

    # The per-timestep targets passed by scan to one_step_loss below will have
    # no leading time axis. We need a treedef without the time axis to use
    # inside one_step_loss to unflatten it back into a dataset:
    # 한글 주석: 아래의 one_step_loss로 scan에 의해 전달되는 시간 단계별 대상은 선행 시간 축이 없습니다.
    # one_step_loss 내부에서 데이터셋으로 다시 평탄화하기 위해 시간 축이 없는 treedef가 필요합니다.
    flat_targets, target_treedef = _get_flat_arrays_and_single_timestep_treedef(
        targets)
    scan_variables = flat_targets
    # 한글 주석: 대상 변수를 평탄화하고 스캔 변수로 설정합니다.

    flat_forcings, forcings_treedef = (
        _get_flat_arrays_and_single_timestep_treedef(forcings))
    scan_variables = (flat_targets, flat_forcings)
    # 한글 주석: 강제 변수를 평탄화하고 대상 변수와 함께 스캔 변수로 설정합니다.

    def one_step_loss(inputs, scan_variables):
      """단일 시간 단계의 손실을 계산하는 내부 함수입니다."""
      # 한글 주석: 이 함수는 hk.scan에 의해 반복적으로 호출되어 각 시간 단계의 손실을 계산합니다.
      flat_target, flat_forcings = scan_variables
      forcings = _unflatten_and_expand_time(flat_forcings, forcings_treedef,
                                            targets.coords['time'][:1])
      # 한글 주석: 스캔 변수에서 평탄화된 강제 변수를 가져와 시간 차원을 복원합니다.

      target = _unflatten_and_expand_time(flat_target, target_treedef,
                                          targets.coords['time'][:1])
      # 한글 주석: 스캔 변수에서 평탄화된 대상 변수를 가져와 시간 차원을 복원합니다.

      # Add constant inputs:
      # 한글 주석: 정적 입력을 추가합니다.
      all_inputs = xarray.merge([constant_inputs, inputs])

      (loss, diagnostics), predictions = self._predictor.loss_and_predictions(
          all_inputs,
          target,
          forcings=forcings,
          **kwargs)
      # 한글 주석: 내부 예측기의 loss_and_predictions 메서드를 호출하여 손실, 진단 정보, 예측을 가져옵니다.

      # Unwrap to jax arrays shape (batch,):
      # 한글 주석: (batch,) 형태의 jax 배열로 래핑 해제합니다.
      loss, diagnostics = xarray_tree.map_structure(
          xarray_jax.unwrap_data, (loss, diagnostics))
      # 한글 주석: 손실과 진단 정보를 xarray에서 일반 JAX 배열로 변환합니다.

      predictions = cast(xarray.Dataset, predictions)  # Keeps pytype happy. # 한글 주석: pytype 검사기를 만족시키기 위해 predictions를 xarray.Dataset으로 캐스팅합니다.
      next_frame = xarray.merge([predictions, forcings])
      # 한글 주석: 예측과 강제 변수를 병합하여 다음 프레임을 만듭니다.
      next_inputs = self._update_inputs(inputs, next_frame)
      # 한글 주석: 다음 반복을 위한 입력을 업데이트합니다.

      return next_inputs, (loss, diagnostics)

    if self._gradient_checkpointing:
      # 한글 주석: 그래디언트 체크포인팅이 활성화된 경우
      scan_length = targets.dims['time']
      if scan_length <= 1:
        logging.warning(
            '길이가 1인 시퀀스에 대해서는 그래디언트 체크포인팅을 건너<0xEB><0xA9><0x95>니다.')
      else:
        one_step_loss = hk.remat(one_step_loss)
        # 한글 주석: hk.remat을 사용하여 그래디언트 체크포인팅을 적용합니다.

    # We can pass inputs (the initial state of the loop) in directly as a
    # Dataset because the shape we pass in to scan is the same as the shape scan
    # passes to the inner function. But, for scan_variables, we must flatten the
    # targets (and unflatten them inside the inner function) because they are
    # passed to the inner function per-timestep without the original time axis.
    # The same apply to the optional forcing.
    # 한글 주석: 입력(루프의 초기 상태)은 scan에 전달하는 모양과 scan이 내부 함수에 전달하는 모양이 같기 때문에
    # Dataset으로 직접 전달할 수 있습니다. 그러나 scan_variables의 경우, 대상(및 선택적 강제 변수)은 원래 시간 축 없이
    # 시간 단계별로 내부 함수에 전달되므로 평탄화해야 합니다 (그리고 내부 함수 내에서 평탄화 해제).
    _, (per_timestep_losses, per_timestep_diagnostics) = hk.scan(
        one_step_loss, inputs, scan_variables)
    # 한글 주석: hk.scan을 사용하여 여러 시간 단계에 걸쳐 one_step_loss를 반복 실행하여 시간 단계별 손실과 진단 정보를 얻습니다.

    # Re-wrap loss and diagnostics as DataArray and average them over time:
    # 한글 주석: 손실과 진단 정보를 DataArray로 다시 래핑하고 시간에 대해 평균을 계산합니다.
    (loss, diagnostics) = jax.tree_util.tree_map(
        lambda x: xarray_jax.DataArray(x, dims=('time', 'batch')).mean(  # pylint: disable=g-long-lambda
            'time', skipna=False),
        (per_timestep_losses, per_timestep_diagnostics))

    return loss, diagnostics
