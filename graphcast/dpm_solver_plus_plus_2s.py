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
"""DPM-Solver++ 2S sampler from https://arxiv.org/abs/2211.01095."""
# 한글 주석: https://arxiv.org/abs/2211.01095 논문의 DPM-Solver++ 2S 샘플러 구현입니다.

from typing import Optional # 한글 주석: 타입 힌트를 위한 모듈 임포트

from graphcast import casting # 데이터 타입 캐스팅 관련 모듈
from graphcast import denoisers_base # 노이즈 제거기 기본 클래스
from graphcast import samplers_base as base # 샘플러 기본 클래스
from graphcast import samplers_utils as utils # 샘플러 유틸리티 함수
from graphcast import xarray_jax # Xarray와 JAX 연동 유틸리티
import haiku as hk # Haiku: JAX를 위한 신경망 라이브러리
import jax.numpy as jnp
import xarray


class Sampler(base.Sampler):
  """Sampling using DPM-Solver++ 2S from [1].
  # 한글 주석: [1]의 DPM-Solver++ 2S를 사용한 샘플링 클래스입니다.

  This is combined with optional stochastic churn as described in [2].
  # 한글 주석: 이는 [2]에 설명된 선택적 확률적 처닝(stochastic churn)과 결합됩니다.

  The '2S' terminology from [1] means that this is a second-order (2),
  single-step (S) solver. Here 'single-step' here distinguishes it from
  'multi-step' methods where the results of function evaluations from previous
  steps are reused in computing updates for subsequent steps. The solver still
  uses multiple steps though.
  # 한글 주석: [1]의 '2S' 용어는 이것이 2차(second-order), 단일 단계(single-step) 솔버임을 의미합니다.
  # 여기서 '단일 단계'는 이전 단계의 함수 평가 결과를 후속 단계의 업데이트 계산에 재사용하는 '다중 단계' 방법과 구별됩니다.
  # 그러나 솔버는 여전히 여러 단계를 사용합니다.

  [1] DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic
  Models, https://arxiv.org/abs/2211.01095
  [2] Elucidating the Design Space of Diffusion-Based Generative Models,
  https://arxiv.org/abs/2206.00364
  """

  def __init__(self,
               denoiser: denoisers_base.Denoiser, # 노이즈 제거기 객체
               max_noise_level: float, # 최대 노이즈 레벨
               min_noise_level: float, # 최소 노이즈 레벨
               num_noise_levels: int, # 노이즈 레벨 수 (샘플링 스텝 수 결정)
               rho: float, # 노이즈 스텝 간격 조절 파라미터
               stochastic_churn_rate: float, # 확률적 처닝 비율
               churn_min_noise_level: float, # 처닝 최소 노이즈 레벨
               churn_max_noise_level: float, # 처닝 최대 노이즈 레벨
               noise_level_inflation_factor: float # 노이즈 레벨 증가 인자
               ):
    """Initializes the sampler.
    # 한글 주석: 샘플러를 초기화합니다.

    Args:
    # 한글 주석: 인수
      denoiser: A Denoiser which predicts noise-free targets.
      # 한글 주석: denoiser: 노이즈 없는 목표를 예측하는 노이즈 제거기입니다.
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
      churn_min_noise_level: Minimum noise level at which stochastic churn
        occurs. S_min from the paper. Only used if stochastic_churn_rate > 0.
      # 한글 주석: churn_min_noise_level: 확률적 처닝이 발생하는 최소 노이즈 레벨입니다. 논문의 S_min입니다. stochastic_churn_rate > 0인 경우에만 사용됩니다.
      churn_max_noise_level: Maximum noise level at which stochastic churn
        occurs. S_min from the paper. Only used if stochastic_churn_rate > 0.
      # 한글 주석: churn_max_noise_level: 확률적 처닝이 발생하는 최대 노이즈 레벨입니다. 논문의 S_max입니다. stochastic_churn_rate > 0인 경우에만 사용됩니다. (주: 원본 주석 S_min은 S_max의 오타로 보임)
      noise_level_inflation_factor: This can be used to set the actual amount of
        noise injected higher than what the denoiser is told has been added.
        The motivation is to compensate for a tendency of L2-trained denoisers
        to remove slightly too much noise / blur too much. S_noise from the
        paper. Only used if stochastic_churn_rate > 0.
      # 한글 주석: noise_level_inflation_factor: 노이즈 제거기에 추가되었다고 알려진 것보다 실제 주입되는 노이즈 양을 더 높게 설정하는 데 사용할 수 있습니다.
      # L2로 학습된 노이즈 제거기가 노이즈를 약간 너무 많이 제거하거나 너무 많이 흐리게 하는 경향을 보상하기 위한 것입니다. 논문의 S_noise입니다. stochastic_churn_rate > 0인 경우에만 사용됩니다.
    """
    super().__init__(denoiser)
    # 노이즈 스케줄 생성
    self._noise_levels = utils.noise_schedule(
        max_noise_level, min_noise_level, num_noise_levels, rho)
    self._stochastic_churn = stochastic_churn_rate > 0 # 확률적 처닝 사용 여부
    # 단계별 처닝 비율 스케줄 생성
    self._per_step_churn_rates = utils.stochastic_churn_rate_schedule(
        self._noise_levels, stochastic_churn_rate, churn_min_noise_level,
        churn_max_noise_level)
    self._noise_level_inflation_factor = noise_level_inflation_factor

  def __call__(
      self,
      inputs: xarray.Dataset, # 입력 데이터셋
      targets_template: xarray.Dataset, # 목표 템플릿 (형태 참조용)
      forcings: Optional[xarray.Dataset] = None, # 선택적 강제 변수
      **kwargs) -> xarray.Dataset: # 기타 키워드 인수
    # 한글 주석: 샘플러를 호출하여 샘플링을 수행합니다.

    dtype = casting.infer_floating_dtype(targets_template)  # pytype: disable=wrong-arg-types # 목표 템플릿에서 부동소수점 타입 추론
    noise_levels = jnp.array(self._noise_levels).astype(dtype) # 노이즈 레벨을 JAX 배열 및 추론된 타입으로 변환
    per_step_churn_rates = jnp.array(self._per_step_churn_rates).astype(dtype) # 단계별 처닝 비율도 변환

    def denoiser(noise_level: jnp.ndarray, x: xarray.Dataset) -> xarray.Dataset:
      """Computes D(x, sigma, y)."""
      # 한글 주석: D(x, sigma, y)를 계산합니다. 즉, 주어진 노이즈 레벨에서 노이즈가 낀 x로부터 노이즈 없는 x0을 예측합니다.
      # 노이즈 레벨을 배치 크기에 맞게 브로드캐스팅합니다.
      bcast_noise_level = xarray_jax.DataArray(
          jnp.tile(noise_level, x.sizes['batch']), dims=('batch',))
      # Estimate the expectation of the fully-denoised target x0, conditional on
      # inputs/forcings, noisy targets and their noise level:
      # 한글 주석: 입력/강제 변수, 노이즈 낀 목표 및 해당 노이즈 레벨을 조건으로 하는 완전히 노이즈 제거된 목표 x0의 기댓값을 추정합니다.
      return self._denoiser( # 내부 노이즈 제거기 모델 호출
          inputs=inputs,
          noisy_targets=x,
          noise_levels=bcast_noise_level,
          forcings=forcings)

    def body_fn(i: jnp.ndarray, x: xarray.Dataset) -> xarray.Dataset:
      """One iteration of the sampling algorithm.
      # 한글 주석: 샘플링 알고리즘의 한 반복입니다.

      Args:
      # 한글 주석: 인수
        i: Sampling iteration.
        # 한글 주석: i: 샘플링 반복 횟수입니다.
        x: Noisy targets at iteration i, these will have noise level
          self._noise_levels[i].
        # 한글 주석: x: 반복 i에서의 노이즈 낀 목표이며, self._noise_levels[i]의 노이즈 레벨을 갖습니다.

      Returns:
      # 한글 주석: 반환값
        Noisy targets at the next lowest noise level self._noise_levels[i+1].
        # 한글 주석: 다음으로 낮은 노이즈 레벨 self._noise_levels[i+1]에서의 노이즈 낀 목표입니다.
      """
      def init_noise(template):
        # 한글 주석: 초기 노이즈를 생성하는 함수입니다. 구형 백색 노이즈를 사용합니다.
        return noise_levels[0] * utils.spherical_white_noise_like(template)

      # Initialise the inputs if i == 0.
      # This is done here to ensure both noise sampler calls can use the same
      # spherical harmonic basis functions. While there may be a small compute
      # cost the memory savings can be significant.
      # TODO(dominicmasters): Figure out if we can merge the two noise sampler
      # calls into one to avoid this hack.
      # 한글 주석: i == 0인 경우 입력을 초기화합니다.
      # 이는 두 노이즈 샘플러 호출이 동일한 구면 조화 함수 기저 함수를 사용할 수 있도록 하기 위함입니다.
      # 약간의 계산 비용이 들 수 있지만 메모리 절약 효과가 클 수 있습니다.
      # TODO(dominicmasters): 이 핵을 피하기 위해 두 노이즈 샘플러 호출을 하나로 병합할 수 있는지 알아봅니다.
      maybe_init_noise = (i == 0).astype(noise_levels[0].dtype) # 첫 번째 스텝에서만 초기 노이즈 적용
      x = x + init_noise(x) * maybe_init_noise # x에 초기 노이즈 추가

      noise_level = noise_levels[i] # 현재 노이즈 레벨

      if self._stochastic_churn: # 확률적 처닝 사용 시
        # We increase the noise level of x a bit before taking it down again:
        # 한글 주석: x의 노이즈 레벨을 약간 높였다가 다시 낮춥니다.
        x, noise_level = utils.apply_stochastic_churn(
            x, noise_level,
            stochastic_churn_rate=per_step_churn_rates[i], # 현재 단계의 처닝 비율
            noise_level_inflation_factor=self._noise_level_inflation_factor) # 노이즈 증가 인자

      # Apply one step of the ODE solver to take x down to the next lowest
      # noise level.
      # 한글 주석: ODE 솔버의 한 단계를 적용하여 x를 다음으로 낮은 노이즈 레벨로 낮춥니다.

      # Note that the Elucidating paper's choice of sigma(t)=t and s(t)=1
      # (corresponding to alpha(t)=1 in the DPM paper) as well as the standard
      # choice of r=1/2 (corresponding to a geometric mean for the s_i
      # midpoints) greatly simplifies the update from the DPM-Solver++ paper.
      # You need to do a bit of algebraic fiddling to arrive at the below after
      # substituting these choices into DPMSolver++'s Algorithm 1. The simpler
      # update we arrive at helps with intuition too.
      # 한글 주석: Elucidating 논문의 sigma(t)=t 및 s(t)=1 선택(DPM 논문의 alpha(t)=1에 해당)과
      # r=1/2의 표준 선택(s_i 중간점에 대한 기하 평균에 해당)은 DPM-Solver++ 논문의 업데이트를 크게 단순화합니다.
      # 이러한 선택을 DPMSolver++의 알고리즘 1에 대입한 후 아래에 도달하려면 약간의 대수적 조작이 필요합니다.
      # 우리가 도달하는 더 간단한 업데이트는 직관에도 도움이 됩니다.

      next_noise_level = noise_levels[i + 1] # 다음 노이즈 레벨
      # This is s_{i+1} from the paper. They don't explain how the s_i are
      # chosen, but the default choice seems to be a geometric mean, which is
      # equivalent to setting all the r_i = 1/2.
      # 한글 주석: 이것은 논문의 s_{i+1}입니다. s_i가 어떻게 선택되는지는 설명하지 않지만,
      # 기본 선택은 기하 평균인 것으로 보이며, 이는 모든 r_i = 1/2로 설정하는 것과 동일합니다.
      mid_noise_level = jnp.sqrt(noise_level * next_noise_level) # 중간 노이즈 레벨 (기하 평균)

      mid_over_current = mid_noise_level / noise_level # 중간 노이즈 레벨 / 현재 노이즈 레벨 비율
      x_denoised = denoiser(noise_level, x) # 현재 노이즈 레벨에서 x를 노이즈 제거
      # This turns out to be a convex combination of current and denoised x,
      # which isn't entirely apparent from the paper formulae:
      # 한글 주석: 이는 현재 x와 노이즈 제거된 x의 볼록 조합(convex combination)으로 나타납니다.
      # 이는 논문 공식에서는 완전히 명확하지 않습니다.
      x_mid = mid_over_current * x + (1 - mid_over_current) * x_denoised # 중간 단계 x 계산

      next_over_current = next_noise_level / noise_level # 다음 노이즈 레벨 / 현재 노이즈 레벨 비율
      x_mid_denoised = denoiser(mid_noise_level, x_mid)  # pytype: disable=wrong-arg-types # 중간 단계 x를 노이즈 제거
      x_next = next_over_current * x + (1 - next_over_current) * x_mid_denoised # 다음 단계 x 계산

      # For the final step to noise level 0, we do an Euler update which
      # corresponds to just returning the denoiser's prediction directly.
      #
      # In fact the behaviour above when next_noise_level == 0 is almost
      # equivalent, except that it runs the denoiser a second time to denoise
      # from noise level 0. The denoiser should just be the identity function in
      # this case, but it hasn't necessarily been trained at noise level 0 so
      # we avoid relying on this.
      # 한글 주석: 노이즈 레벨 0으로 가는 마지막 단계에서는 오일러 업데이트를 수행하며, 이는 노이즈 제거기의 예측을 직접 반환하는 것에 해당합니다.
      # 사실 next_noise_level == 0일 때 위의 동작은 거의 동일하지만, 노이즈 레벨 0에서 노이즈를 제거하기 위해 노이즈 제거기를 두 번 실행한다는 점이 다릅니다.
      # 이 경우 노이즈 제거기는 항등 함수여야 하지만, 반드시 노이즈 레벨 0에서 학습된 것은 아니므로 이에 의존하지 않습니다.
      return utils.tree_where(next_noise_level == 0, x_denoised, x_next) # 다음 노이즈 레벨이 0이면 x_denoised, 아니면 x_next 반환

    # Init with zeros but apply additional noise at step 0 to initialise the
    # state.
    # 한글 주석: 0으로 초기화하지만 상태를 초기화하기 위해 0단계에서 추가 노이즈를 적용합니다.
    noise_init = xarray.zeros_like(targets_template) # 목표 템플릿과 동일한 형태의 0으로 채워진 배열로 초기화
    # hk.fori_loop를 사용하여 샘플링 반복 수행
    return hk.fori_loop(
        0, len(noise_levels) - 1, body_fun=body_fn, init_val=noise_init)
