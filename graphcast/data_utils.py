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
"""Dataset utilities."""
# 한글 주석: 데이터셋 관련 유틸리티 함수들을 모아놓은 모듈입니다.

from typing import Any, Mapping, Sequence, Tuple, Union # 한글 주석: 타입 힌트를 위한 모듈 임포트

from graphcast import solar_radiation # 한글 주석: 태양 복사 관련 모듈 임포트
import numpy as np
import pandas as pd
import xarray

TimedeltaLike = Any  # Something convertible to pd.Timedelta.
# 한글 주석: pd.Timedelta로 변환 가능한 모든 타입을 나타내는 별칭입니다.
TimedeltaStr = str  # A string convertible to pd.Timedelta.
# 한글 주석: pd.Timedelta로 변환 가능한 문자열을 나타내는 별칭입니다.

TargetLeadTimes = Union[
    TimedeltaLike, # 단일 리드 타임
    Sequence[TimedeltaLike], # 리드 타임 시퀀스
    slice  # with TimedeltaLike as its start and stop. # 리드 타임 슬라이스
]
# 한글 주석: 목표 리드 타임을 나타내는 타입 정의입니다. 단일 값, 시퀀스, 또는 슬라이스 형태를 가질 수 있습니다.

_SEC_PER_HOUR = 3600 # 시간당 초
_HOUR_PER_DAY = 24 # 하루당 시간
SEC_PER_DAY = _SEC_PER_HOUR * _HOUR_PER_DAY # 하루당 초
_AVG_DAY_PER_YEAR = 365.24219 # 연 평균 일수 (윤년 고려)
AVG_SEC_PER_YEAR = SEC_PER_DAY * _AVG_DAY_PER_YEAR # 연 평균 초

DAY_PROGRESS = "day_progress" # 일 진행률 변수명
YEAR_PROGRESS = "year_progress" # 연 진행률 변수명
_DERIVED_VARS = { # 파생 변수 집합
    DAY_PROGRESS,
    f"{DAY_PROGRESS}_sin", # 일 진행률의 sin 값
    f"{DAY_PROGRESS}_cos", # 일 진행률의 cos 값
    YEAR_PROGRESS,
    f"{YEAR_PROGRESS}_sin", # 연 진행률의 sin 값
    f"{YEAR_PROGRESS}_cos", # 연 진행률의 cos 값
}
TISR = "toa_incident_solar_radiation" # 대기 상층 입사 태양 복사량 변수명


def get_year_progress(seconds_since_epoch: np.ndarray) -> np.ndarray:
  """Computes year progress for times in seconds.
  # 한글 주석: 초 단위 시간으로부터 연 진행률을 계산합니다.

  Args:
  # 한글 주석: 인수
    seconds_since_epoch: Times in seconds since the "epoch" (the point at which
      UNIX time starts).
    # 한글 주석: seconds_since_epoch: "에포크"(UNIX 시간이 시작되는 시점) 이후의 시간을 초 단위로 나타낸 배열입니다.

  Returns:
  # 한글 주석: 반환값
    Year progress normalized to be in the [0, 1) interval for each time point.
    # 한글 주석: 각 시점에 대해 [0, 1) 구간으로 정규화된 연 진행률입니다.
  """

  # Start with the pure integer division, and then float at the very end.
  # We will try to keep as much precision as possible.
  # 한글 주석: 순수 정수 나눗셈으로 시작하여 맨 마지막에 부동소수점으로 변환합니다.
  # 가능한 한 많은 정밀도를 유지하려고 합니다.
  years_since_epoch = (
      seconds_since_epoch / SEC_PER_DAY / np.float64(_AVG_DAY_PER_YEAR)
  )
  # Note depending on how these ops are down, we may end up with a "weak_type"
  # which can cause issues in subtle ways, and hard to track here.
  # In any case, casting to float32 should get rid of the weak type.
  # [0, 1.) Interval.
  # 한글 주석: 이러한 연산 방식에 따라 "weak_type"이 발생하여 미묘한 문제를 일으키고 추적하기 어려울 수 있습니다.
  # 어떤 경우든 float32로 캐스팅하면 weak_type이 제거됩니다.
  # [0, 1) 구간입니다.
  return np.mod(years_since_epoch, 1.0).astype(np.float32)


def get_day_progress(
    seconds_since_epoch: np.ndarray,
    longitude: np.ndarray,
) -> np.ndarray:
  """Computes day progress for times in seconds at each longitude.
  # 한글 주석: 각 경도에서 초 단위 시간으로부터 일 진행률을 계산합니다.

  Args:
  # 한글 주석: 인수
    seconds_since_epoch: 1D array of times in seconds since the 'epoch' (the
      point at which UNIX time starts).
    # 한글 주석: seconds_since_epoch: '에포크'(UNIX 시간이 시작되는 시점) 이후의 시간을 초 단위로 나타낸 1D 배열입니다.
    longitude: 1D array of longitudes at which day progress is computed.
    # 한글 주석: longitude: 일 진행률이 계산될 경도를 나타내는 1D 배열입니다.

  Returns:
  # 한글 주석: 반환값
    2D array of day progress values normalized to be in the [0, 1) inverval
      for each time point at each longitude.
    # 한글 주석: 각 경도의 각 시점에 대해 [0, 1) 구간으로 정규화된 일 진행률 값을 나타내는 2D 배열입니다.
  """

  # [0.0, 1.0) Interval.
  # 한글 주석: [0.0, 1.0) 구간입니다.
  day_progress_greenwich = (
      np.mod(seconds_since_epoch, SEC_PER_DAY) / SEC_PER_DAY # 그리니치 표준시 기준 일 진행률
  )

  # Offset the day progress to the longitude of each point on Earth.
  # 한글 주석: 지구 각 지점의 경도에 맞게 일 진행률을 오프셋합니다.
  longitude_offsets = np.deg2rad(longitude) / (2 * np.pi) # 경도를 라디안으로 변환 후 [0,1) 범위로 정규화
  day_progress = np.mod(
      day_progress_greenwich[..., np.newaxis] + longitude_offsets, 1.0 # 그리니치 기준 진행률에 경도 오프셋 더함
  )
  return day_progress.astype(np.float32)


def featurize_progress(
    name: str, dims: Sequence[str], progress: np.ndarray
) -> Mapping[str, xarray.Variable]:
  """Derives features used by ML models from the `progress` variable.
  # 한글 주석: `progress` 변수로부터 ML 모델에 사용될 특징들을 파생시킵니다.

  Args:
  # 한글 주석: 인수
    name: Base variable name from which features are derived.
    # 한글 주석: name: 특징이 파생될 기본 변수 이름입니다.
    dims: List of the output feature dimensions, e.g. ("day", "lon").
    # 한글 주석: dims: 출력 특징 차원의 리스트입니다. 예: ("day", "lon").
    progress: Progress variable values.
    # 한글 주석: progress: 진행률 변수 값입니다.

  Returns:
  # 한글 주석: 반환값
    Dictionary of xarray variables derived from the `progress` values. It
    includes the original `progress` variable along with its sin and cos
    transformations.
    # 한글 주석: `progress` 값에서 파생된 xarray 변수들의 딕셔너리입니다.
    # 원래 `progress` 변수와 함께 sin 및 cos 변환된 값을 포함합니다.

  Raises:
  # 한글 주석: 발생 오류
    ValueError if the number of feature dimensions is not equal to the number
      of data dimensions.
    # 한글 주석: ValueError: 특징 차원의 수가 데이터 차원의 수와 같지 않은 경우 발생합니다.
  """
  if len(dims) != progress.ndim:
    raise ValueError(
        f"특징 차원의 수({len(dims)})는 데이터 차원의 수와 같아야 합니다: {progress.ndim}."
    )
  progress_phase = progress * (2 * np.pi) # 진행률을 [0, 2*pi) 범위로 변환 (위상)
  return {
      name: xarray.Variable(dims, progress), # 원본 진행률
      name + "_sin": xarray.Variable(dims, np.sin(progress_phase)), # sin 변환
      name + "_cos": xarray.Variable(dims, np.cos(progress_phase)), # cos 변환
  }


def get_seconds_since_epoch(datetime_sequence: xarray.DataArray) -> np.ndarray:
  """Computes seconds since epoch from `data` in place if missing."""
  # 한글 주석: `data`에서 에포크 이후 초를 계산합니다 (없는 경우).
  # Note `datetime_sequence.astype("datetime64[s]").astype(np.int64)`
  # does not work as xarrays always cast dates into nanoseconds!
  # 한글 주석: `datetime_sequence.astype("datetime64[s]").astype(np.int64)`는
  # xarray가 항상 날짜를 나노초로 캐스팅하기 때문에 작동하지 않습니다!
  return datetime_sequence.data.astype("datetime64[s]").astype(np.int64)


def add_derived_vars(data: xarray.Dataset) -> None:
  """Adds year and day progress features to `data` in place if missing.
  # 한글 주석: `data`에 연 및 일 진행률 특징을 추가합니다 (없는 경우).

  Args:
  # 한글 주석: 인수
    data: Xarray dataset to which derived features will be added.
    # 한글 주석: data: 파생 특징이 추가될 Xarray 데이터셋입니다.

  Raises:
  # 한글 주석: 발생 오류
    ValueError if `datetime` or `lon` are not in `data` coordinates.
    # 한글 주석: ValueError: `datetime` 또는 `lon`이 `data` 좌표에 없는 경우 발생합니다.
  """

  for coord in ("datetime", "lon"):
    if coord not in data.coords:
      raise ValueError(f"'{coord}'는 `data` 좌표에 있어야 합니다.")

  # Compute seconds since epoch.
  # 한글 주석: 에포크 이후 초를 계산합니다.
  seconds_since_epoch = get_seconds_since_epoch(data.coords["datetime"])
  batch_dim = ("batch",) if "batch" in data.dims else () # 배치 차원 존재 여부 확인

  # Add year progress features if missing.
  # 한글 주석: 연 진행률 특징이 없는 경우 추가합니다.
  if YEAR_PROGRESS not in data.data_vars:
    year_progress = get_year_progress(seconds_since_epoch)
    data.update( # 데이터셋에 파생된 연 진행률 특징들 업데이트
        featurize_progress(
            name=YEAR_PROGRESS,
            dims=batch_dim + ("time",),
            progress=year_progress,
        )
    )

  # Add day progress features if missing.
  # 한글 주석: 일 진행률 특징이 없는 경우 추가합니다.
  if DAY_PROGRESS not in data.data_vars:
    longitude_coord = data.coords["lon"]
    day_progress = get_day_progress(seconds_since_epoch, longitude_coord.data)
    data.update( # 데이터셋에 파생된 일 진행률 특징들 업데이트
        featurize_progress(
            name=DAY_PROGRESS,
            dims=batch_dim + ("time",) + longitude_coord.dims,
            progress=day_progress,
        )
    )


def add_tisr_var(data: xarray.Dataset) -> None:
  """Adds TISR feature to `data` in place if missing.
  # 한글 주석: `data`에 TISR(대기 상층 입사 태양 복사량) 특징을 추가합니다 (없는 경우).

  Args:
  # 한글 주석: 인수
    data: Xarray dataset to which TISR feature will be added.
    # 한글 주석: data: TISR 특징이 추가될 Xarray 데이터셋입니다.

  Raises:
  # 한글 주석: 발생 오류
    ValueError if `datetime`, 'lat', or `lon` are not in `data` coordinates.
    # 한글 주석: ValueError: `datetime`, 'lat', 또는 `lon`이 `data` 좌표에 없는 경우 발생합니다.
  """

  if TISR in data.data_vars: # 이미 TISR 변수가 있으면 반환
    return

  for coord in ("datetime", "lat", "lon"):
    if coord not in data.coords:
      raise ValueError(f"'{coord}'는 `data` 좌표에 있어야 합니다.")

  # Remove `batch` dimension of size one if present. An error will be raised if
  # the `batch` dimension exists and has size greater than one.
  # 한글 주석: `batch` 차원이 존재하고 크기가 1이면 제거합니다.
  # `batch` 차원이 존재하고 크기가 1보다 크면 오류가 발생합니다.
  data_no_batch = data.squeeze("batch") if "batch" in data.dims else data

  tisr = solar_radiation.get_toa_incident_solar_radiation_for_xarray(
      data_no_batch, use_jit=True # JIT 컴파일 사용하여 TISR 계산
  )

  if "batch" in data.dims: # 원래 배치 차원이 있었으면 다시 추가
    tisr = tisr.expand_dims("batch", axis=0)

  data.update({TISR: tisr}) # 데이터셋에 TISR 변수 업데이트


def extract_input_target_times(
    dataset: xarray.Dataset,
    input_duration: TimedeltaLike,
    target_lead_times: TargetLeadTimes,
    ) -> Tuple[xarray.Dataset, xarray.Dataset]:
  """Extracts inputs and targets for prediction, from a Dataset with a time dim.
  # 한글 주석: 시간 차원을 가진 데이터셋에서 예측을 위한 입력과 목표를 추출합니다.

  The input period is assumed to be contiguous (specified by a duration), but
  the targets can be a list of arbitrary lead times.
  # 한글 주석: 입력 기간은 연속적이라고 가정하지만(기간으로 지정), 목표는 임의의 리드 타임 목록일 수 있습니다.

  Examples:
  # 한글 주석: 예시:

    # Use 18 hours of data as inputs, and two specific lead times as targets:
    # 3 days and 5 days after the final input.
    # 한글 주석: 18시간 데이터를 입력으로 사용하고, 최종 입력 후 3일 및 5일 뒤의 두 특정 리드 타임을 목표로 사용합니다.
    extract_inputs_targets(
        dataset,
        input_duration='18h',
        target_lead_times=('3d', '5d')
    )

    # Use 1 day of data as input, and all lead times between 6 hours and
    # 24 hours inclusive as targets. Demonstrates a friendlier supported string
    # syntax.
    # 한글 주석: 1일 데이터를 입력으로 사용하고, 6시간에서 24시간 사이(포함)의 모든 리드 타임을 목표로 사용합니다.
    # 더 친숙하게 지원되는 문자열 구문을 보여줍니다.
    extract_inputs_targets(
        dataset,
        input_duration='1 day',
        target_lead_times=slice('6 hours', '24 hours')
    )

    # Just use a single target lead time of 3 days:
    # 한글 주석: 3일의 단일 목표 리드 타임만 사용합니다.
    extract_inputs_targets(
        dataset,
        input_duration='24h',
        target_lead_times='3d'
    )

  Args:
  # 한글 주석: 인수
    dataset: An xarray.Dataset with a 'time' dimension whose coordinates are
      timedeltas. It's assumed that the time coordinates have a fixed offset /
      time resolution, and that the input_duration and target_lead_times are
      multiples of this.
    # 한글 주석: dataset: 시간 좌표가 timedelta인 'time' 차원을 가진 xarray.Dataset입니다.
    # 시간 좌표는 고정된 오프셋/시간 해상도를 가지며, input_duration 및 target_lead_times는 이의 배수라고 가정합니다.
    input_duration: pandas.Timedelta or something convertible to it (e.g. a
      shorthand string like '6h' or '5d12h').
    # 한글 주석: input_duration: pandas.Timedelta 또는 이로 변환 가능한 것 (예: '6h' 또는 '5d12h'와 같은 약식 문자열).
    target_lead_times: Either a single lead time, a slice with start and stop
      (inclusive) lead times, or a sequence of lead times. Lead times should be
      Timedeltas (or something convertible to). They are given relative to the
      final input timestep, and should be positive.
    # 한글 주석: target_lead_times: 단일 리드 타임, 시작과 끝(포함)이 있는 슬라이스 또는 리드 타임 시퀀스입니다.
    # 리드 타임은 Timedelta (또는 변환 가능한 것)여야 합니다. 최종 입력 시간 단계에 상대적으로 주어지며 양수여야 합니다.

  Returns:
  # 한글 주석: 반환값
    inputs:
    # 한글 주석: inputs: 입력 데이터셋입니다.
    targets:
    # 한글 주석: targets: 목표 데이터셋입니다.
      Two datasets with the same shape as the input dataset except that a
      selection has been made from the time axis, and the origin of the
      time coordinate will be shifted to refer to lead times relative to the
      final input timestep. So for inputs the times will end at lead time 0,
      for targets the time coordinates will refer to the lead times requested.
    # 한글 주석: 입력 데이터셋과 동일한 모양을 가지지만 시간 축에서 선택이 이루어지고 시간 좌표의 원점이
    # 최종 입력 시간 단계에 상대적인 리드 타임을 참조하도록 이동된 두 개의 데이터셋입니다.
    # 따라서 입력의 경우 시간은 리드 타임 0에서 끝나고, 목표의 경우 시간 좌표는 요청된 리드 타임을 참조합니다.
  """

  (target_lead_times, target_duration
   ) = _process_target_lead_times_and_get_duration(target_lead_times)
  # 한글 주석: 목표 리드 타임을 처리하고 최대 목표 기간을 가져옵니다.

  # Shift the coordinates for the time axis so that a timedelta of zero
  # corresponds to the forecast reference time. That is, the final timestep
  # that's available as input to the forecast, with all following timesteps
  # forming the target period which needs to be predicted.
  # This means the time coordinates are now forecast lead times.
  # 한글 주석: 시간 축의 좌표를 이동하여 시간차 0이 예측 기준 시간에 해당하도록 합니다.
  # 즉, 예측에 입력으로 사용할 수 있는 최종 시간 단계이며, 이후 모든 시간 단계는 예측해야 할 목표 기간을 형성합니다.
  # 이는 시간 좌표가 이제 예측 리드 타임임을 의미합니다.
  time = dataset.coords["time"]
  dataset = dataset.assign_coords(time=time + target_duration - time[-1]) # 시간 좌표를 예측 기준 시간 기준으로 변경

  # Slice out targets:
  # 한글 주석: 목표를 슬라이싱합니다.
  targets = dataset.sel({"time": target_lead_times})

  input_duration = pd.Timedelta(input_duration)
  # Both endpoints are inclusive with label-based slicing, so we offset by a
  # small epsilon to make one of the endpoints non-inclusive:
  # 한글 주석: 레이블 기반 슬라이싱에서는 양쪽 끝점이 모두 포함되므로, 작은 엡실론만큼 오프셋하여
  # 한쪽 끝점을 포함하지 않도록 만듭니다.
  zero = pd.Timedelta(0)
  epsilon = pd.Timedelta(1, "ns") # 매우 작은 시간 간격
  inputs = dataset.sel({"time": slice(-input_duration + epsilon, zero)}) # 입력 기간 슬라이싱
  return inputs, targets


def _process_target_lead_times_and_get_duration(
    target_lead_times: TargetLeadTimes) -> TimedeltaLike:
  """Returns the minimum duration for the target lead times."""
  # 한글 주석: 목표 리드 타임에 대한 최소 기간을 반환합니다. (실제로는 최대 리드 타임을 반환하여 전체 기간을 커버)
  if isinstance(target_lead_times, slice):
    # A slice of lead times. xarray already accepts timedelta-like values for
    # the begin/end/step of the slice.
    # 한글 주석: 리드 타임의 슬라이스입니다. xarray는 이미 슬라이스의 시작/끝/단계에 대해 timedelta 유사 값을 허용합니다.
    if target_lead_times.start is None:
      # If the start isn't specified, we assume it starts at the next timestep
      # after lead time 0 (lead time 0 is the final input timestep):
      # 한글 주석: 시작이 지정되지 않은 경우, 리드 타임 0(최종 입력 시간 단계) 이후의 다음 시간 단계에서 시작한다고 가정합니다.
      target_lead_times = slice(
          pd.Timedelta(1, "ns"), target_lead_times.stop, target_lead_times.step
      )
    target_duration = pd.Timedelta(target_lead_times.stop) # 슬라이스의 끝을 전체 기간으로 사용
  else:
    if not isinstance(target_lead_times, (list, tuple, set)):
      # A single lead time, which we wrap as a length-1 array to ensure there
      # still remains a time dimension (here of length 1) for consistency.
      # 한글 주석: 단일 리드 타임인 경우, 일관성을 위해 길이가 1인 배열로 래핑하여 시간 차원이 유지되도록 합니다.
      target_lead_times = [target_lead_times]

    # A list of multiple (not necessarily contiguous) lead times:
    # 한글 주석: 여러 (반드시 연속적이지 않은) 리드 타임 목록입니다.
    target_lead_times = [pd.Timedelta(x) for x in target_lead_times] # 모든 리드 타임을 Timedelta로 변환
    target_lead_times.sort() # 정렬
    target_duration = target_lead_times[-1] # 가장 긴 리드 타임을 전체 기간으로 사용
  return target_lead_times, target_duration


def extract_inputs_targets_forcings(
    dataset: xarray.Dataset,
    *,
    input_variables: Tuple[str, ...],
    target_variables: Tuple[str, ...],
    forcing_variables: Tuple[str, ...],
    pressure_levels: Tuple[int, ...],
    input_duration: TimedeltaLike,
    target_lead_times: TargetLeadTimes,
    ) -> Tuple[xarray.Dataset, xarray.Dataset, xarray.Dataset]:
  """Extracts inputs, targets and forcings according to requirements."""
  # 한글 주석: 요구 사항에 따라 입력, 목표 및 강제 변수를 추출합니다.
  dataset = dataset.sel(level=list(pressure_levels)) # 지정된 기압 수준 선택

  # "Forcings" include derived variables that do not exist in the original ERA5
  # or HRES datasets, as well as other variables (e.g. tisr) that need to be
  # computed manually for the target lead times. Compute the requested ones.
  # 한글 주석: "강제 변수"에는 원래 ERA5 또는 HRES 데이터셋에 존재하지 않는 파생 변수와
  # 목표 리드 타임에 대해 수동으로 계산해야 하는 기타 변수(예: tisr)가 포함됩니다. 요청된 변수를 계산합니다.
  if set(forcing_variables) & _DERIVED_VARS: # 요청된 강제 변수 중에 파생 변수가 포함되어 있으면
    add_derived_vars(dataset) # 파생 변수 추가
  if set(forcing_variables) & {TISR}: # 요청된 강제 변수 중에 TISR이 포함되어 있으면
    add_tisr_var(dataset) # TISR 변수 추가

  # `datetime` is needed by add_derived_vars but breaks autoregressive rollouts.
  # 한글 주석: `datetime`은 add_derived_vars에 필요하지만 자동 회귀 롤아웃을 방해합니다.
  dataset = dataset.drop_vars("datetime") # datetime 변수 제거

  inputs, targets = extract_input_target_times(
      dataset,
      input_duration=input_duration,
      target_lead_times=target_lead_times)
  # 한글 주석: 입력 및 목표 시간 기준으로 데이터셋 분리

  if set(forcing_variables) & set(target_variables): # 강제 변수와 목표 변수가 겹치는지 확인
    raise ValueError(
        f"강제 변수 {forcing_variables}는 목표 변수 {target_variables}와 겹치지 않아야 합니다."
    )

  inputs = inputs[list(input_variables)] # 지정된 입력 변수만 선택
  # The forcing uses the same time coordinates as the target.
  # 한글 주석: 강제 변수는 목표와 동일한 시간 좌표를 사용합니다.
  forcings = targets[list(forcing_variables)] # 목표 시간대의 강제 변수 선택
  targets = targets[list(target_variables)] # 지정된 목표 변수만 선택

  return inputs, targets, forcings
