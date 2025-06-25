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
"""Wrappers that take care of casting."""
# 한글 주석: 데이터 타입 캐스팅(형변환)을 처리하는 래퍼(wrapper) 모듈입니다.

import contextlib
from typing import Any, Mapping, Tuple

import chex
from graphcast import predictor_base
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import xarray


PyTree = Any
# 한글 주석: 임의의 중첩된 JAX 데이터 구조 (PyTree)를 나타내는 타입 별칭입니다.


class Bfloat16Cast(predictor_base.Predictor):
  """Wrapper that casts all inputs to bfloat16 and outputs to targets dtype."""
  # 한글 주석: 모든 입력을 bfloat16으로 캐스팅하고 출력은 대상(targets)의 데이터 타입으로 캐스팅하는 래퍼 클래스입니다.

  def __init__(self, predictor: predictor_base.Predictor, enabled: bool = True):
    """Inits the wrapper.
    # 한글 주석: 래퍼를 초기화합니다.

    Args:
    # 한글 주석: 인수
      predictor: predictor being wrapped.
      # 한글 주석: predictor: 래핑될 예측기 객체입니다.
      enabled: disables the wrapper if False, for simpler hyperparameter scans.
      # 한글 주석: enabled: False일 경우 래퍼를 비활성화하여 하이퍼파라미터 검색을 단순화합니다.
    """
    self._enabled = enabled
    self._predictor = predictor

  def __call__(self,
               inputs: xarray.Dataset,
               targets_template: xarray.Dataset,
               forcings: xarray.Dataset,
               **kwargs
               ) -> xarray.Dataset:
    # 한글 주석: 예측기를 호출하여 예측을 수행합니다.
    if not self._enabled:
      # 한글 주석: 래퍼가 비활성화된 경우, 내부 예측기를 직접 호출합니다.
      return self._predictor(inputs, targets_template, forcings, **kwargs)

    with bfloat16_variable_view():
      # 한글 주석: bfloat16 변수 뷰 컨텍스트 내에서 실행합니다.
      # 이 컨텍스트는 Haiku 모듈의 파라미터는 float32로 유지하면서 활성화 값(activation)은 bfloat16으로 처리하도록 합니다.
      predictions = self._predictor(
          *_all_inputs_to_bfloat16(inputs, targets_template, forcings), # 모든 입력을 bfloat16으로 변환
          **kwargs,)

    predictions_dtype = infer_floating_dtype(predictions)  # pytype: disable=wrong-arg-types # 한글 주석: 예측 결과의 부동 소수점 데이터 타입을 추론합니다.
    if predictions_dtype != jnp.bfloat16:
      # 한글 주석: 예측 결과가 bfloat16이 아니면 오류를 발생시킵니다.
      raise ValueError(f'Expected bfloat16 output, got {predictions_dtype}')

    targets_dtype = infer_floating_dtype(targets_template)  # pytype: disable=wrong-arg-types # 한글 주석: 대상 템플릿의 부동 소수점 데이터 타입을 추론합니다.
    # 한글 주석: 예측 결과를 대상 데이터 타입으로 캐스팅하여 반환합니다.
    return tree_map_cast(
        predictions, input_dtype=jnp.bfloat16, output_dtype=targets_dtype)

  def loss(self,
           inputs: xarray.Dataset,
           targets: xarray.Dataset,
           forcings: xarray.Dataset,
           **kwargs,
           ) -> predictor_base.LossAndDiagnostics:
    # 한글 주석: 손실(loss) 및 진단 정보(diagnostics)를 계산합니다.
    if not self._enabled:
      # 한글 주석: 래퍼가 비활성화된 경우, 내부 예측기의 loss 메서드를 직접 호출합니다.
      return self._predictor.loss(inputs, targets, forcings, **kwargs)

    with bfloat16_variable_view():
      # 한글 주석: bfloat16 변수 뷰 컨텍스트 내에서 실행합니다.
      loss, scalars = self._predictor.loss(
          *_all_inputs_to_bfloat16(inputs, targets, forcings), **kwargs) # 모든 입력을 bfloat16으로 변환

    if loss.dtype != jnp.bfloat16:
      # 한글 주석: 계산된 손실이 bfloat16이 아니면 오류를 발생시킵니다.
      raise ValueError(f'Expected bfloat16 loss, got {loss.dtype}')

    targets_dtype = infer_floating_dtype(targets)  # pytype: disable=wrong-arg-types # 한글 주석: 대상의 부동 소수점 데이터 타입을 추론합니다.

    # Note that casting back the loss to e.g. float32 should not affect data
    # types of the backwards pass, because the first thing the backwards pass
    # should do is to go backwards the casting op and cast back to bfloat16
    # (and xprofs seem to confirm this).
    # 한글 주석: 손실을 예를 들어 float32로 다시 캐스팅하는 것은 역전파(backwards pass)의 데이터 타입에 영향을 미치지 않아야 합니다.
    # 왜냐하면 역전파가 가장 먼저 수행하는 작업은 캐스팅 연산을 거슬러 올라가 bfloat16으로 다시 캐스팅하는 것이기 때문입니다
    # (xprof 결과도 이를 확인시켜 줍니다).
    # 한글 주석: 손실과 스칼라 값들을 대상 데이터 타입으로 캐스팅하여 반환합니다.
    return tree_map_cast((loss, scalars),
                         input_dtype=jnp.bfloat16, output_dtype=targets_dtype)

  def loss_and_predictions(  # pytype: disable=signature-mismatch  # jax-ndarray
      self,
      inputs: xarray.Dataset,
      targets: xarray.Dataset,
      forcings: xarray.Dataset,
      **kwargs,
      ) -> Tuple[predictor_base.LossAndDiagnostics,
                 xarray.Dataset]:
    # 한글 주석: 손실, 진단 정보 및 예측 결과를 함께 계산하여 반환합니다.
    if not self._enabled:
      # 한글 주석: 래퍼가 비활성화된 경우, 내부 예측기의 loss_and_predictions 메서드를 직접 호출합니다.
      return self._predictor.loss_and_predictions(inputs, targets, forcings,  # pytype: disable=bad-return-type  # jax-ndarray
                                                  **kwargs)

    with bfloat16_variable_view():
      # 한글 주석: bfloat16 변수 뷰 컨텍스트 내에서 실행합니다.
      (loss, scalars), predictions = self._predictor.loss_and_predictions(
          *_all_inputs_to_bfloat16(inputs, targets, forcings), **kwargs) # 모든 입력을 bfloat16으로 변환

    if loss.dtype != jnp.bfloat16:
      # 한글 주석: 계산된 손실이 bfloat16이 아니면 오류를 발생시킵니다.
      raise ValueError(f'Expected bfloat16 loss, got {loss.dtype}')

    predictions_dtype = infer_floating_dtype(predictions)  # pytype: disable=wrong-arg-types # 한글 주석: 예측 결과의 부동 소수점 데이터 타입을 추론합니다.
    if predictions_dtype != jnp.bfloat16:
      # 한글 주석: 예측 결과가 bfloat16이 아니면 오류를 발생시킵니다.
      raise ValueError(f'Expected bfloat16 output, got {predictions_dtype}')

    targets_dtype = infer_floating_dtype(targets)  # pytype: disable=wrong-arg-types # 한글 주석: 대상의 부동 소수점 데이터 타입을 추론합니다.
    # 한글 주석: 손실, 스칼라 값, 예측 결과를 대상 데이터 타입으로 캐스팅하여 반환합니다.
    return tree_map_cast(((loss, scalars), predictions),
                         input_dtype=jnp.bfloat16, output_dtype=targets_dtype)


def infer_floating_dtype(data_vars: Mapping[str, chex.Array]) -> np.dtype:
  """Infers a floating dtype from an input mapping of data."""
  # 한글 주석: 데이터 매핑 입력으로부터 부동 소수점 데이터 타입을 추론합니다.
  # 데이터 변수들 중에서 부동 소수점 타입인 것들의 집합을 만듭니다.
  dtypes = {
      v.dtype
      for k, v in data_vars.items() if jnp.issubdtype(v.dtype, np.floating)}
  if len(dtypes) != 1:
    # 한글 주석: 부동 소수점 타입이 정확히 하나가 아니면 오류를 발생시킵니다.
    dtypes_and_shapes = {
        k: (v.dtype, v.shape)
        for k, v in data_vars.items() if jnp.issubdtype(v.dtype, np.floating)}
    raise ValueError(
        f'입력 변수에서 정확히 하나의 부동 소수점 dtype {dtypes}을(를) 찾지 못했습니다: '
        f'{dtypes_and_shapes}')
  return list(dtypes)[0] # 한글 주석: 추론된 단일 부동 소수점 데이터 타입을 반환합니다.


def _all_inputs_to_bfloat16(
    inputs: xarray.Dataset,
    targets: xarray.Dataset,
    forcings: xarray.Dataset,
    ) -> Tuple[xarray.Dataset,
               xarray.Dataset,
               xarray.Dataset]:
  """Converts all input datasets to bfloat16 data type."""
  # 한글 주석: 모든 입력 데이터셋(inputs, targets, forcings)을 bfloat16 데이터 타입으로 변환합니다.
  return (inputs.astype(jnp.bfloat16),
          jax.tree.map(lambda x: x.astype(jnp.bfloat16), targets), # targets는 PyTree 구조일 수 있으므로 tree.map 사용
          forcings.astype(jnp.bfloat16))


def tree_map_cast(inputs: PyTree, input_dtype: np.dtype, output_dtype: np.dtype,
                  ) -> PyTree:
  """Casts elements of a PyTree from input_dtype to output_dtype if they match input_dtype."""
  # 한글 주석: PyTree의 요소들을 input_dtype과 일치하는 경우 output_dtype으로 캐스팅합니다.
  def cast_fn(x):
    # 한글 주석: 입력 요소 x의 데이터 타입이 input_dtype과 같으면 output_dtype으로 캐스팅합니다.
    if x.dtype == input_dtype:
      return x.astype(output_dtype)
  return jax.tree.map(cast_fn, inputs) # PyTree의 모든 요소에 cast_fn 함수를 적용합니다.


@contextlib.contextmanager
def bfloat16_variable_view(enabled: bool = True):
  """Context for Haiku modules with float32 params, but bfloat16 activations.
  # 한글 주석: float32 파라미터를 사용하지만 bfloat16 활성화를 사용하는 Haiku 모듈을 위한 컨텍스트입니다.

  It works as follows:
  # 한글 주석: 다음과 같이 작동합니다:
  * Every time a variable is requested to be created/set as np.bfloat16,
    it will create an underlying float32 variable, instead.
  # 한글 주석: * 변수가 np.bfloat16으로 생성/설정되도록 요청될 때마다 대신 기본 float32 변수를 생성합니다.
  * Every time a variable a variable is requested as bfloat16, it will check the
    variable is of float32 type, and cast the variable to bfloat16.
  # 한글 주석: * 변수가 bfloat16으로 요청될 때마다 변수가 float32 타입인지 확인하고 변수를 bfloat16으로 캐스팅합니다.

  Note the gradients are still computed and accumulated as float32, because
  the params returned by init are float32, so the gradient function with
  respect to the params will already include an implicit casting to float32.
  # 한글 주석: init에 의해 반환된 파라미터가 float32이기 때문에 그래디언트는 여전히 float32로 계산되고 누적됩니다.
  # 따라서 파라미터에 대한 그래디언트 함수에는 이미 float32로의 암시적 캐스팅이 포함됩니다.

  Args:
  # 한글 주석: 인수
    enabled: Only enables bfloat16 behavior if True.
    # 한글 주석: enabled: True인 경우에만 bfloat16 동작을 활성화합니다.

  Yields:
  # 한글 주석: 반환 (생성자)
    None
  """

  if enabled:
    # 한글 주석: bfloat16 동작이 활성화된 경우, Haiku의 custom_creator, custom_getter, custom_setter를 사용하여
    # 변수 생성, 조회, 설정을 커스터마이징합니다.
    with hk.custom_creator(
        _bfloat16_creator, state=True), hk.custom_getter(
            _bfloat16_getter, state=True), hk.custom_setter(
                _bfloat16_setter):
      yield
  else:
    # 한글 주석: 비활성화된 경우, 아무 작업도 하지 않고 컨텍스트를 종료합니다.
    yield


def _bfloat16_creator(next_creator, shape, dtype, init, context):
  """Creates float32 variables when bfloat16 is requested."""
  # 한글 주석: bfloat16이 요청될 때 float32 변수를 생성합니다.
  # 원래 요청된 dtype이 bfloat16이면, 실제로는 float32로 생성합니다.
  if context.original_dtype == jnp.bfloat16:
    dtype = jnp.float32
  return next_creator(shape, dtype, init)


def _bfloat16_getter(next_getter, value, context):
  """Casts float32 to bfloat16 when bfloat16 was originally requested."""
  # 한글 주석: 원래 bfloat16이 요청되었을 때 float32를 bfloat16으로 캐스팅합니다.
  # 조회된 변수의 원래 요청 타입이 bfloat16이고 실제 저장된 타입이 float32이면, bfloat16으로 캐스팅하여 반환합니다.
  if context.original_dtype == jnp.bfloat16:
    assert value.dtype == jnp.float32 # 실제 저장된 타입이 float32인지 확인
    value = value.astype(jnp.bfloat16)
  return next_getter(value)


def _bfloat16_setter(next_setter, value, context):
  """Casts bfloat16 to float32 when bfloat16 was originally set."""
  # 한글 주석: 원래 bfloat16으로 설정되었을 때 bfloat16을 float32로 캐스팅합니다.
  # 설정하려는 값의 원래 요청 타입이 bfloat16이면, 실제로는 float32로 캐스팅하여 저장합니다.
  if context.original_dtype == jnp.bfloat16:
    value = value.astype(jnp.float32)
  return next_setter(value)
