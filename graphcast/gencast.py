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
"""Denoising diffusion models based on the framework of [1].
# 한글 주석: [1]의 프레임워크를 기반으로 하는 노이즈 제거 확산 모델입니다.

Throughout we will refer to notation and equations from [1].
# 한글 주석: 전체적으로 [1]의 표기법과 방정식을 참조합니다.

  [1] Elucidating the Design Space of Diffusion-Based Generative Models
  Karras, Aittala, Aila and Laine, 2022
  https://arxiv.org/abs/2206.00364
"""

from typing import Any, Optional, Tuple # 한글 주석: 타입 힌트를 위한 모듈 임포트

import chex # JAX 및 NumPy를 위한 유틸리티 라이브러리
from graphcast import casting # 데이터 타입 캐스팅 관련 모듈
from graphcast import denoiser # 노이즈 제거기 모듈
from graphcast import dpm_solver_plus_plus_2s # DPM-Solver++ 2S 샘플러 모듈
from graphcast import graphcast # GraphCast 관련 설정 및 변수
from graphcast import losses # 손실 함수 모듈
from graphcast import predictor_base # 예측기 기본 클래스
from graphcast import samplers_utils # 샘플러 유틸리티 함수
from graphcast import xarray_jax # Xarray와 JAX 연동 유틸리티
import haiku as hk # Haiku: JAX를 위한 신경망 라이브러리
import jax
import xarray


TARGET_SURFACE_VARS = ( # GenCast 모델의 목표 지표면 변수들
    '2m_temperature', # 2m 온도
    'mean_sea_level_pressure', # 평균 해수면 기압
    '10m_v_component_of_wind', # 10m 바람의 V 성분
    '10m_u_component_of_wind',  # GenCast predicts in 12hr timesteps. # 10m 바람의 U 성분 (GenCast는 12시간 간격으로 예측)
    'total_precipitation_12hr', # 12시간 총 강수량
    'sea_surface_temperature', # 해수면 온도
)

TARGET_SURFACE_NO_PRECIP_VARS = ( # 강수량을 제외한 목표 지표면 변수들 (입력으로 사용)
    '2m_temperature',
    'mean_sea_level_pressure',
    '10m_v_component_of_wind',
    '10m_u_component_of_wind',
    'sea_surface_temperature',
)


TASK = graphcast.TaskConfig( # GenCast 모델의 작업 구성 정의
    input_variables=( # 입력 변수들
        # GenCast doesn't take precipitation as input.
        # 한글 주석: GenCast는 강수량을 입력으로 받지 않습니다.
        TARGET_SURFACE_NO_PRECIP_VARS # 강수량 제외 지표면 변수
        + graphcast.TARGET_ATMOSPHERIC_VARS # GraphCast의 대기 변수
        + graphcast.GENERATED_FORCING_VARS # GraphCast의 생성된 강제 변수 (시간 진행률 등)
        + graphcast.STATIC_VARS # GraphCast의 정적 변수 (지표 고도 등)
    ),
    target_variables=TARGET_SURFACE_VARS + graphcast.TARGET_ATMOSPHERIC_VARS, # 목표 변수 (지표면 + 대기)
    # GenCast doesn't take incident solar radiation as a forcing.
    # 한글 주석: GenCast는 입사 태양 복사를 강제 변수로 받지 않습니다.
    forcing_variables=graphcast.GENERATED_FORCING_VARS, # 생성된 강제 변수만 사용
    pressure_levels=graphcast.PRESSURE_LEVELS_WEATHERBENCH_13, # 기압 수준 (WeatherBench 13 레벨)
    # GenCast takes the current frame and the frame 12 hours prior.
    # 한글 주석: GenCast는 현재 프레임과 12시간 전 프레임을 입력으로 받습니다. (총 24시간)
    input_duration='24h',
)


@chex.dataclass(frozen=True, eq=True)
class SamplerConfig:
  """Configures the sampler used to draw samples from GenCast.
  # 한글 주석: GenCast에서 샘플을 추출하는 데 사용되는 샘플러를 구성합니다.

      max_noise_level: The highest noise level used at the start of the
        sequence of reverse diffusion steps.
      # 한글 주석: max_noise_level: 역 확산 단계 시퀀스의 시작 부분에 사용되는 가장 높은 노이즈 레벨입니다.
      min_noise_level: The lowest noise level used at the end of the sequence of
        reverse diffusion steps.
      # 한글 주석: min_noise_level: 역 확산 단계 시퀀스의 끝 부분에 사용되는 가장 낮은 노이즈 레벨입니다.
      num_noise_levels: Determines the number of noise levels used and hence the
        number of reverse diffusion steps performed.
      # 한글 주석: num_noise_levels: 사용되는 노이즈 레벨의 수를 결정하며, 따라서 수행되는 역 확산 단계의 수를 결정합니다.
      rho: Parameter affecting the spacing of noise steps. Higher values will
        concentrate noise steps more around zero.
      # 한글 주석: rho: 노이즈 단계의 간격에 영향을 미치는 매개변수입니다. 값이 높을수록 노이즈 단계가 0 주위에 더 집중됩니다.
      stochastic_churn_rate: S_churn from the paper. This controls the rate
        at which noise is re-injected/'churned' during the sampling algorithm.
        If this is set to zero then we are performing deterministic sampling
        as described in Algorithm 1.
      # 한글 주석: stochastic_churn_rate: 논문의 S_churn입니다. 샘플링 알고리즘 동안 노이즈가 다시 주입되거나 '처닝'되는 비율을 제어합니다.
      # 이 값이 0이면 알고리즘 1에 설명된 대로 결정론적 샘플링을 수행합니다.
      churn_max_noise_level: Maximum noise level at which stochastic churn
        occurs. S_min from the paper. Only used if stochastic_churn_rate > 0.
      # 한글 주석: churn_max_noise_level: 확률적 처닝이 발생하는 최대 노이즈 레벨입니다. 논문의 S_max입니다. stochastic_churn_rate > 0인 경우에만 사용됩니다. (주: 원본 주석 S_min은 S_max의 오타로 보임)
      churn_min_noise_level: Minimum noise level at which stochastic churn
        occurs. S_min from the paper. Only used if stochastic_churn_rate > 0.
      # 한글 주석: churn_min_noise_level: 확률적 처닝이 발생하는 최소 노이즈 레벨입니다. 논문의 S_min입니다. stochastic_churn_rate > 0인 경우에만 사용됩니다.
      noise_level_inflation_factor: This can be used to set the actual amount of
        noise injected higher than what the denoiser is told has been added.
        The motivation is to compensate for a tendency of L2-trained denoisers
        to remove slightly too much noise / blur too much. S_noise from the
        paper. Only used if stochastic_churn_rate > 0.
      # 한글 주석: noise_level_inflation_factor: 노이즈 제거기에 추가되었다고 알려진 것보다 실제 주입되는 노이즈 양을 더 높게 설정하는 데 사용할 수 있습니다.
      # L2로 학습된 노이즈 제거기가 노이즈를 약간 너무 많이 제거하거나 너무 많이 흐리게 하는 경향을 보상하기 위한 것입니다. 논문의 S_noise입니다. stochastic_churn_rate > 0인 경우에만 사용됩니다.
  """
  max_noise_level: float = 80.
  min_noise_level: float = 0.03
  num_noise_levels: int = 20
  rho: float = 7.
  # Stochastic sampler settings.
  # 한글 주석: 확률적 샘플러 설정입니다.
  stochastic_churn_rate: float = 2.5
  churn_min_noise_level: float = 0.75
  churn_max_noise_level: float = float('inf') # 무한대
  noise_level_inflation_factor: float = 1.05


@chex.dataclass(frozen=True, eq=True)
class NoiseConfig:
  # 한글 주석: 학습 시 사용되는 노이즈 관련 설정을 정의합니다.
  training_noise_level_rho: float = 7.0 # 학습용 노이즈 레벨 rho 값
  training_max_noise_level: float = 88.0 # 학습용 최대 노이즈 레벨
  training_min_noise_level: float = 0.02 # 학습용 최소 노이즈 레벨


@chex.dataclass(frozen=True, eq=True)
class CheckPoint:
  # 한글 주석: 모델 체크포인트 저장 및 로드를 위한 데이터 구조입니다.
  description: str # 설명
  license: str # 라이선스
  params: dict[str, Any] # 모델 파라미터
  task_config: graphcast.TaskConfig # 작업 설정
  denoiser_architecture_config: denoiser.DenoiserArchitectureConfig # 노이즈 제거기 아키텍처 설정
  sampler_config: SamplerConfig # 샘플러 설정
  noise_config: NoiseConfig # 노이즈 설정
  noise_encoder_config: denoiser.NoiseEncoderConfig # 노이즈 인코더 설정


class GenCast(predictor_base.Predictor):
  """Predictor for a denoising diffusion model following the framework of [1].
  # 한글 주석: [1]의 프레임워크를 따르는 노이즈 제거 확산 모델을 위한 예측기입니다.

    [1] Elucidating the Design Space of Diffusion-Based Generative Models
    Karras, Aittala, Aila and Laine, 2022
    https://arxiv.org/abs/2206.00364

  Unlike the paper, we have a conditional model and our denoising function
  conditions on previous timesteps.
  # 한글 주석: 논문과 달리, 우리는 조건부 모델을 사용하며 노이즈 제거 함수는 이전 시간 단계에 의존합니다.

  As the paper demonstrates, the sampling algorithm can be varied independently
  of the denoising model and its training procedure, and it is separately
  configurable here.
  # 한글 주석: 논문에서 보여주듯이, 샘플링 알고리즘은 노이즈 제거 모델 및 학습 절차와 독립적으로 변경될 수 있으며, 여기서는 별도로 구성 가능합니다.
  """

  def __init__(
      self,
      task_config: graphcast.TaskConfig, # 작업 설정
      denoiser_architecture_config: denoiser.DenoiserArchitectureConfig, # 노이즈 제거기 아키텍처 설정
      sampler_config: Optional[SamplerConfig] = None, # 샘플러 설정 (추론 시 필요)
      noise_config: Optional[NoiseConfig] = None, # 노이즈 설정 (학습 시 필요)
      noise_encoder_config: Optional[denoiser.NoiseEncoderConfig] = None, # 노이즈 인코더 설정
  ):
    """Constructs GenCast."""
    # 한글 주석: GenCast 모델을 구성합니다.
    # Output size depends on number of variables being predicted.
    # 한글 주석: 출력 크기는 예측되는 변수의 수에 따라 달라집니다.
    num_surface_vars = len( # 지표면 변수 수 계산
        set(task_config.target_variables)
        - set(graphcast.ALL_ATMOSPHERIC_VARS)
    )
    num_atmospheric_vars = len( # 대기 변수 수 계산
        set(task_config.target_variables)
        & set(graphcast.ALL_ATMOSPHERIC_VARS)
    )
    num_outputs = ( # 총 출력 수 계산
        num_surface_vars
        + len(task_config.pressure_levels) * num_atmospheric_vars
    )
    # 노이즈 제거기 아키텍처 설정에 노드 출력 크기 설정
    denoiser_architecture_config.node_output_size = num_outputs
    # Denoiser 객체 생성
    self._denoiser = denoiser.Denoiser(
        noise_encoder_config,
        denoiser_architecture_config,
    )
    self._sampler_config = sampler_config # 샘플러 설정 저장
    # Singleton to avoid re-initializing the sampler for each inference call.
    # 한글 주석: 각 추론 호출 시 샘플러를 다시 초기화하지 않도록 하기 위한 싱글톤입니다.
    self._sampler = None # 샘플러 객체 (추론 시 초기화됨)
    self._noise_config = noise_config # 노이즈 설정 저장

  def _c_in(self, noise_scale: xarray.DataArray) -> xarray.DataArray:
    """Scaling applied to the noisy targets input to the underlying network."""
    # 한글 주석: 기본 네트워크에 입력되는 노이즈 낀 목표에 적용되는 스케일링입니다. (논문의 c_in)
    return (noise_scale**2 + 1)**-0.5

  def _c_out(self, noise_scale: xarray.DataArray) -> xarray.DataArray:
    """Scaling applied to the underlying network's raw outputs."""
    # 한글 주석: 기본 네트워크의 원시 출력에 적용되는 스케일링입니다. (논문의 c_out)
    return noise_scale * (noise_scale**2 + 1)**-0.5

  def _c_skip(self, noise_scale: xarray.DataArray) -> xarray.DataArray:
    """Scaling applied to the skip connection."""
    # 한글 주석: 스킵 연결에 적용되는 스케일링입니다. (논문의 c_skip)
    return 1 / (noise_scale**2 + 1)

  def _loss_weighting(self, noise_scale: xarray.DataArray) -> xarray.DataArray:
    r"""The loss weighting \lambda(\sigma) from the paper."""
    # 한글 주석: 논문의 손실 가중치 \lambda(\sigma) 입니다.
    return self._c_out(noise_scale) ** -2

  def _preconditioned_denoiser(
      self,
      inputs: xarray.Dataset,
      noisy_targets: xarray.Dataset,
      noise_levels: xarray.DataArray,
      forcings: Optional[xarray.Dataset] = None,
      **kwargs) -> xarray.Dataset:
    """The preconditioned denoising function D from the paper (Eqn 7)."""
    # 한글 주석: 논문의 사전 조건화된 노이즈 제거 함수 D (방정식 7)입니다.
    # 내부 노이즈 제거기(_denoiser)를 호출하여 원시 예측을 얻습니다.
    # 이때 입력은 c_in 스케일링된 노이즈 낀 목표입니다.
    raw_predictions = self._denoiser(
        inputs=inputs,
        noisy_targets=noisy_targets * self._c_in(noise_levels), # c_in 적용
        noise_levels=noise_levels,
        forcings=forcings,
        **kwargs)
    # 사전 조건화 공식에 따라 최종 노이즈 제거된 예측을 계산합니다.
    return (raw_predictions * self._c_out(noise_levels) + # c_out 적용
            noisy_targets * self._c_skip(noise_levels)) # c_skip 적용

  def loss_and_predictions(
      self,
      inputs: xarray.Dataset,
      targets: xarray.Dataset,
      forcings: Optional[xarray.Dataset] = None,
  ) -> Tuple[predictor_base.LossAndDiagnostics, xarray.Dataset]:
    # 한글 주석: 손실과 예측을 함께 반환합니다. (주로 학습 중에 사용될 수 있음)
    return self.loss(inputs, targets, forcings), self(inputs, targets, forcings)

  def loss(self,
           inputs: xarray.Dataset,
           targets: xarray.Dataset,
           forcings: Optional[xarray.Dataset] = None,
           ) -> predictor_base.LossAndDiagnostics:
    # 한글 주석: 모델의 손실을 계산합니다.

    if self._noise_config is None: # 학습 시 노이즈 설정이 없으면 오류 발생
      raise ValueError('GenCast 학습을 위해서는 노이즈 설정을 지정해야 합니다.')

    # Sample noise levels:
    # 한글 주석: 노이즈 레벨을 샘플링합니다.
    dtype = casting.infer_floating_dtype(targets)  # pytype: disable=wrong-arg-types # 목표 데이터로부터 부동소수점 타입 추론
    key = hk.next_rng_key() # Haiku RNG 키 생성
    batch_size = inputs.sizes['batch'] # 배치 크기
    # 지정된 분포(rho_inverse_cdf)에 따라 노이즈 레벨 샘플링
    noise_levels = xarray_jax.DataArray(
        data=samplers_utils.rho_inverse_cdf(
            min_value=self._noise_config.training_min_noise_level,
            max_value=self._noise_config.training_max_noise_level,
            rho=self._noise_config.training_noise_level_rho,
            cdf=jax.random.uniform(key, shape=(batch_size,), dtype=dtype)), # 균등 분포에서 CDF 값 샘플링
        dims=('batch',))

    # Sample noise and apply it to targets:
    # 한글 주석: 노이즈를 샘플링하여 목표에 적용합니다.
    noise = ( # 목표와 동일한 형태의 구형 백색 노이즈 생성 후 노이즈 레벨 곱하기
        samplers_utils.spherical_white_noise_like(targets) * noise_levels
    )
    noisy_targets = targets + noise # 원본 목표에 노이즈 추가

    # 사전 조건화된 노이즈 제거 함수를 사용하여 예측 수행
    denoised_predictions = self._preconditioned_denoiser(
        inputs, noisy_targets, noise_levels, forcings)

    # 가중 MSE 손실 계산
    loss, diagnostics = losses.weighted_mse_per_level(
        denoised_predictions,
        targets,
        # Weights are same as we used for GraphCast.
        # 한글 주석: 가중치는 GraphCast에서 사용한 것과 동일합니다.
        per_variable_weights={ # 변수별 가중치
            # Any variables not specified here are weighted as 1.0.
            # 한글 주석: 여기에 지정되지 않은 변수는 1.0으로 가중치가 부여됩니다.
            # A single-level variable, but an important headline variable
            # and also one which we have struggled to get good performance
            # on at short lead times, so leaving it weighted at 1.0, equal
            # to the multi-level variables:
            # 한글 주석: 단일 레벨 변수이지만 중요한 주요 변수이며 짧은 리드 타임에서 좋은 성능을 얻기 어려웠으므로
            # 다중 레벨 변수와 동일하게 1.0으로 가중치를 유지합니다.
            '2m_temperature': 1.0,
            # New single-level variables, which we don't weight too highly
            # to avoid hurting performance on other variables.
            # 한글 주석: 다른 변수의 성능을 저해하지 않도록 너무 높게 가중치를 부여하지 않는 새로운 단일 레벨 변수입니다.
            '10m_u_component_of_wind': 0.1,
            '10m_v_component_of_wind': 0.1,
            'mean_sea_level_pressure': 0.1,
            'sea_surface_temperature': 0.1,
            'total_precipitation_12hr': 0.1
        },
    )
    loss *= self._loss_weighting(noise_levels) # 최종 손실에 노이즈 레벨에 따른 가중치 적용
    return loss, diagnostics

  def __call__(self,
               inputs: xarray.Dataset,
               targets_template: xarray.Dataset, # 목표 템플릿 (출력 형태 정의용)
               forcings: Optional[xarray.Dataset] = None,
               **kwargs) -> xarray.Dataset:
    # 한글 주석: 모델을 호출하여 추론(샘플링)을 수행합니다.
    if self._sampler_config is None: # 추론 시 샘플러 설정이 없으면 오류 발생
      raise ValueError(
          'GenCast에서 추론을 실행하려면 샘플러 설정을 지정해야 합니다.'
      )
    if self._sampler is None: # 샘플러가 아직 초기화되지 않았으면
      # DPM-Solver++ 2S 샘플러 객체 생성 및 저장
      self._sampler = dpm_solver_plus_plus_2s.Sampler(
          self._preconditioned_denoiser, **self._sampler_config # 사전 조건화된 노이즈 제거 함수와 샘플러 설정 전달
      )
    # 저장된 샘플러를 사용하여 샘플링 수행
    return self._sampler(inputs, targets_template, forcings, **kwargs)
