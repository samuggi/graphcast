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
"""Base class for Denoisers used in diffusion Predictors.
# 한글 주석: 확산(diffusion) 예측기에서 사용되는 노이즈 제거기(Denoiser)의 기본 클래스입니다.

Denoisers are a bit like deterministic Predictors, except:
# 한글 주석: 노이즈 제거기는 결정론적 예측기와 약간 유사하지만 다음과 같은 차이점이 있습니다:
* Their __call__ method also conditions on noisy_targets and the noise_levels
  of those noisy targets
# 한글 주석: * __call__ 메서드는 노이즈가 낀 목표(noisy_targets)와 해당 목표의 노이즈 레벨(noise_levels)에도 의존(조건화)합니다.
* They don't have an overrideable loss function (the loss is assumed to be some
  form of MSE and is implemented outside the Denoiser itself)
# 한글 주석: * 재정의 가능한 손실 함수가 없습니다 (손실은 일종의 MSE로 가정되며 노이즈 제거기 외부에서 구현됩니다).
"""

from typing import Optional, Protocol # 한글 주석: 타입 힌트를 위한 모듈 임포트

import xarray # xarray 라이브러리 임포트


class Denoiser(Protocol):
  """A denoising model that conditions on inputs as well as noise level."""
  # 한글 주석: 입력뿐만 아니라 노이즈 레벨에도 의존하는 노이즈 제거 모델의 프로토콜(인터페이스)입니다.
  # 이 클래스를 직접 상속받는 대신, 이 프로토콜을 따르는 클래스를 구현하여 타입 검사를 활용할 수 있습니다.

  def __call__(
      self,
      inputs: xarray.Dataset, # 입력 데이터셋
      noisy_targets: xarray.Dataset, # 노이즈가 낀 목표 데이터셋
      noise_levels: xarray.DataArray, # 노이즈 레벨 (DataArray 형태)
      forcings: Optional[xarray.Dataset] = None, # 선택적인 추가 강제 변수
      **kwargs) -> xarray.Dataset: # 기타 키워드 인수
    """Computes denoised targets from noisy targets.
    # 한글 주석: 노이즈가 낀 목표로부터 노이즈가 제거된 목표를 계산합니다.

    Args:
    # 한글 주석: 인수
      inputs: Inputs to condition on, as for Predictor.__call__.
      # 한글 주석: inputs: 조건으로 사용할 입력입니다 (Predictor.__call__과 유사).
      noisy_targets: Targets which have had i.i.d. zero-mean Gaussian noise
        added to them (where the noise level used may vary along the 'batch'
        dimension).
      # 한글 주석: noisy_targets: 평균이 0인 독립적이고 동일하게 분포된(i.i.d.) 가우시안 노이즈가 추가된 목표입니다
      # (사용된 노이즈 레벨은 'batch' 차원을 따라 달라질 수 있음).
      noise_levels: A DataArray with dimensions ('batch',) specifying the noise
        levels that were used for each example in the batch.
      # 한글 주석: noise_levels: ('batch',) 차원을 가진 DataArray로, 배치의 각 예제에 사용된 노이즈 레벨을 지정합니다.
      forcings: Optional additional per-target-timestep forcings to condition
        on, as for Predictor.__call__.
      # 한글 주석: forcings: 조건으로 사용할 선택적인 추가 목표 시간 단계별 강제 변수입니다 (Predictor.__call__과 유사).
      **kwargs: Any additional custom kwargs.
      # 한글 주석: **kwargs: 기타 사용자 정의 키워드 인수입니다.

    Returns:
    # 한글 주석: 반환값
      Denoised predictions with the same shape as noisy_targets.
      # 한글 주석: noisy_targets와 동일한 형태를 가진 노이즈가 제거된 예측입니다.
    """
    # 한글 주석: 이 메서드는 프로토콜의 일부이므로 실제 구현은 이 프로토콜을 따르는 구체적인 클래스에서 제공되어야 합니다.
    # 여기에 pass 또는 ... (Ellipsis)를 사용하여 프로토콜 메서드임을 명시할 수 있습니다.
    ...
