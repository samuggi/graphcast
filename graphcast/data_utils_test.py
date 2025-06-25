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
"""Tests for `data_utils.py`."""
# 한글 주석: `data_utils.py` 파일의 함수들에 대한 테스트 코드입니다.

import datetime
from absl.testing import absltest
from absl.testing import parameterized # 한글 주석: 파라미터화된 테스트를 위한 absl 유틸리티 임포트
from graphcast import data_utils # 한글 주석: 테스트 대상 모듈 임포트
import numpy as np
import xarray as xa


class DataUtilsTest(parameterized.TestCase):
  # 한글 주석: data_utils 모듈의 함수들을 테스트하는 클래스입니다.

  def setUp(self):
    # 한글 주석: 각 테스트 메소드 실행 전에 호출됩니다.
    super().setUp()
    # Fix the seed for reproducibility.
    # 한글 주석: 재현성을 위해 난수 시드를 고정합니다.
    np.random.seed(0)

  def test_year_progress_is_zero_at_year_start_or_end(self):
    # 한글 주석: get_year_progress 함수가 연초 또는 연말에 0을 반환하는지 테스트합니다.
    year_progress = data_utils.get_year_progress(
        np.array([
            0, # 에포크 시작
            data_utils.AVG_SEC_PER_YEAR, # 1년 후
            data_utils.AVG_SEC_PER_YEAR * 42,  # 42 years. # 42년 후
        ])
    )
    # 한글 주석: 계산된 연 진행률이 모두 0인지 확인합니다.
    np.testing.assert_array_equal(year_progress, np.zeros(year_progress.shape))

  def test_year_progress_is_almost_one_before_year_ends(self):
    # 한글 주석: get_year_progress 함수가 연말 직전에 거의 1에 가까운 값을 반환하는지 테스트합니다.
    year_progress = data_utils.get_year_progress(
        np.array([
            data_utils.AVG_SEC_PER_YEAR - 1, # 1년 되기 1초 전
            (data_utils.AVG_SEC_PER_YEAR - 1) * 42,  # ~42 years # 약 42년 되기 1초 전 (실제로는 (1년-1초) * 42)
        ])
    )
    with self.subTest("Year progress values are close to 1"):
      # 한글 주석: 연 진행률 값이 1에 가까운지 (0.999보다 큰지) 확인합니다.
      self.assertTrue(np.all(year_progress > 0.999))
    with self.subTest("Year progress values != 1"):
      # 한글 주석: 연 진행률 값이 정확히 1은 아닌지 (1보다 작은지) 확인합니다.
      self.assertTrue(np.all(year_progress < 1.0))

  def test_day_progress_computes_for_all_times_and_longitudes(self):
    # 한글 주석: get_day_progress 함수가 모든 주어진 시간과 경도에 대해 일 진행률을 계산하는지 테스트합니다.
    times = np.random.randint(low=0, high=1e10, size=10) # 임의의 시간 값들
    longitudes = np.arange(0, 360.0, 1.0) # 0도부터 359도까지 1도 간격의 경도 값들
    day_progress = data_utils.get_day_progress(times, longitudes)
    with self.subTest("Day progress is computed for all times and longinutes"):
      # 한글 주석: 계산된 일 진행률 배열의 형태가 (시간 개수, 경도 개수)와 일치하는지 확인합니다.
      self.assertSequenceEqual(
          day_progress.shape, (len(times), len(longitudes))
      )

  @parameterized.named_parameters(
      # 한글 주석: 여러 날짜/시간 조합에 대해 동일한 테스트를 실행하기 위한 파라미터화입니다.
      dict(
          testcase_name="random_date_1", # 테스트 케이스 이름
          year=1988,
          month=11,
          day=7,
          hour=2,
          minute=45,
          second=34,
      ),
      dict(
          testcase_name="random_date_2",
          year=2022,
          month=3,
          day=12,
          hour=7,
          minute=1,
          second=0,
      ),
  )
  def test_day_progress_is_in_between_zero_and_one(
      self, year, month, day, hour, minute, second
  ):
    # 한글 주석: get_day_progress 함수가 반환하는 일 진행률 값이 항상 [0, 1) 범위 내에 있는지 테스트합니다.
    # Datetime from a timestamp.
    # 한글 주석: 타임스탬프로부터 datetime 객체를 생성합니다.
    dt = datetime.datetime(year, month, day, hour, minute, second)
    # Epoch time.
    # 한글 주석: 에포크 시간 (1970년 1월 1일)
    epoch_time = datetime.datetime(1970, 1, 1)
    # Seconds since epoch.
    # 한글 주석: 에포크 이후 초를 계산합니다.
    seconds_since_epoch = np.array([(dt - epoch_time).total_seconds()])

    # Longitudes with 1 degree resolution.
    # 한글 주석: 1도 해상도의 경도 값들입니다.
    longitudes = np.arange(0, 360.0, 1.0)

    day_progress = data_utils.get_day_progress(seconds_since_epoch, longitudes)
    with self.subTest("Day progress >= 0"):
      # 한글 주석: 일 진행률 값이 0보다 크거나 같은지 확인합니다.
      self.assertTrue(np.all(day_progress >= 0.0))
    with self.subTest("Day progress < 1"):
      # 한글 주석: 일 진행률 값이 1보다 작은지 확인합니다.
      self.assertTrue(np.all(day_progress < 1.0))

  def test_day_progress_is_zero_at_day_start_or_end(self):
    # 한글 주석: get_day_progress 함수가 하루의 시작 또는 끝(경도 0 기준)에 0을 반환하는지 테스트합니다.
    day_progress = data_utils.get_day_progress(
        seconds_since_epoch=np.array([
            0, # 에포크 시작 (자정)
            data_utils.SEC_PER_DAY, # 하루 뒤 (자정)
            data_utils.SEC_PER_DAY * 42,  # 42 days. # 42일 뒤 (자정)
        ]),
        longitude=np.array([0.0]), # 경도 0
    )
    # 한글 주석: 계산된 일 진행률이 모두 0인지 확인합니다.
    np.testing.assert_array_equal(day_progress, np.zeros(day_progress.shape))

  def test_day_progress_specific_value(self):
    # 한글 주석: get_day_progress 함수가 특정 입력에 대해 예상되는 정확한 값을 반환하는지 테스트합니다.
    day_progress = data_utils.get_day_progress(
        seconds_since_epoch=np.array([123]), # 123초
        longitude=np.array([0.0]), # 경도 0
    )
    # 123 / (24 * 3600) = 0.0014236111...
    np.testing.assert_array_almost_equal(
        day_progress, np.array([[0.00142361]]), decimal=6 # 소수점 6자리까지 비교
    )

  def test_featurize_progress_valid_values_and_dimensions(self):
    # 한글 주석: featurize_progress 함수가 올바른 값과 차원을 가진 특징들을 생성하는지 테스트합니다.
    day_progress = np.array([0.0, 0.45, 0.213]) # 테스트용 일 진행률 값
    feature_dimensions = ("time",) # 예상되는 특징 차원
    progress_features = data_utils.featurize_progress(
        name="day_progress", dims=feature_dimensions, progress=day_progress
    )
    for feature_name, feature_var in progress_features.items(): # 오타 수정: feature -> feature_var, values() -> items()
      with self.subTest(f"Valid dimensions for {feature_name}"): # 오타 수정: feature -> feature_name
        # 한글 주석: 각 생성된 특징의 차원이 예상과 일치하는지 확인합니다.
        self.assertSequenceEqual(feature_var.dims, feature_dimensions)

    with self.subTest("Valid values for day_progress"):
      # 한글 주석: "day_progress" 특징의 값이 원본 진행률과 일치하는지 확인합니다.
      np.testing.assert_array_equal(
          day_progress, progress_features["day_progress"].values
      )

    with self.subTest("Valid values for day_progress_sin"):
      # 한글 주석: "day_progress_sin" 특징의 값이 올바른 sin 변환 값인지 확인합니다.
      np.testing.assert_array_almost_equal(
          np.array([0.0, np.sin(0.45 * 2 * np.pi), np.sin(0.213 * 2 * np.pi)]), # 직접 계산된 값과 비교
          progress_features["day_progress_sin"].values,
          decimal=6,
      )

    with self.subTest("Valid values for day_progress_cos"):
      # 한글 주석: "day_progress_cos" 특징의 값이 올바른 cos 변환 값인지 확인합니다.
      np.testing.assert_array_almost_equal(
          np.array([1.0, np.cos(0.45 * 2 * np.pi), np.cos(0.213 * 2 * np.pi)]), # 직접 계산된 값과 비교
          progress_features["day_progress_cos"].values,
          decimal=6,
      )

  def test_featurize_progress_invalid_dimensions(self):
    # 한글 주석: featurize_progress 함수에 잘못된 차원이 입력되었을 때 ValueError를 발생하는지 테스트합니다.
    year_progress = np.array([0.0, 0.45, 0.213])
    feature_dimensions = ("time", "longitude") # progress 배열의 차원(1D)과 불일치
    with self.assertRaises(ValueError): # ValueError가 발생하는지 확인
      data_utils.featurize_progress(
          name="year_progress", dims=feature_dimensions, progress=year_progress
      )

  def test_add_derived_vars_variables_added(self):
    # 한글 주석: add_derived_vars 함수가 데이터셋에 필요한 파생 변수들(연/일 진행률 관련)을 올바르게 추가하는지 테스트합니다.
    data = xa.Dataset(
        data_vars={
            "var1": (["x", "lon", "datetime"], 8 * np.random.randn(2, 2, 3))
        },
        coords={
            "lon": np.array([0.0, 0.5]),
            "datetime": np.array([
                datetime.datetime(2021, 1, 1),
                datetime.datetime(2023, 1, 1),
                datetime.datetime(2023, 1, 3),
            ]),
        },
    )
    data_utils.add_derived_vars(data) # 파생 변수 추가 함수 호출
    all_variables = set(data.variables) # 데이터셋의 모든 변수 이름 집합

    with self.subTest("Original value was not removed"):
      # 한글 주석: 기존 변수가 제거되지 않았는지 확인합니다.
      self.assertIn("var1", all_variables)
    with self.subTest("Year progress feature was added"):
      # 한글 주석: 연 진행률 관련 특징들이 추가되었는지 확인합니다.
      self.assertIn(data_utils.YEAR_PROGRESS, all_variables)
      self.assertIn(data_utils.YEAR_PROGRESS + "_sin", all_variables) # sin, cos도 확인
      self.assertIn(data_utils.YEAR_PROGRESS + "_cos", all_variables)
    with self.subTest("Day progress feature was added"):
      # 한글 주석: 일 진행률 관련 특징들이 추가되었는지 확인합니다.
      self.assertIn(data_utils.DAY_PROGRESS, all_variables)
      self.assertIn(data_utils.DAY_PROGRESS + "_sin", all_variables) # sin, cos도 확인
      self.assertIn(data_utils.DAY_PROGRESS + "_cos", all_variables)

  def test_add_derived_vars_existing_vars_not_overridden(self):
    # 한글 주석: add_derived_vars 함수가 이미 존재하는 파생 변수를 덮어쓰지 않는지 테스트합니다.
    dims = ["x", "lon", "datetime"]
    # 미리 정의된 연/일 진행률 값을 가진 데이터셋 생성
    existing_year_progress_val = 0.111
    existing_day_progress_val = 0.222
    data = xa.Dataset(
        data_vars={
            "var1": (dims, 8 * np.random.randn(2, 2, 3)),
            data_utils.YEAR_PROGRESS: (dims[:-1], np.full((2, 2), existing_year_progress_val)), # YEAR_PROGRESS는 datetime 차원 없음
            data_utils.DAY_PROGRESS: (dims, np.full((2, 2, 3), existing_day_progress_val)),
        },
        coords={
            "lon": np.array([0.0, 0.5]),
            "datetime": np.array([
                datetime.datetime(2021, 1, 1),
                datetime.datetime(2023, 1, 1),
                datetime.datetime(2023, 1, 3),
            ]),
        },
    )

    data_utils.add_derived_vars(data) # 파생 변수 추가 함수 호출 (이미 변수 존재)

    with self.subTest("Year progress feature was not overridden"):
      # 한글 주석: YEAR_PROGRESS 변수가 기존 값으로 유지되는지 확인합니다.
      np.testing.assert_allclose(data[data_utils.YEAR_PROGRESS].isel(datetime=0), existing_year_progress_val) # datetime 차원이 없으므로 isel 사용
    with self.subTest("Day progress feature was not overridden"):
      # 한글 주석: DAY_PROGRESS 변수가 기존 값으로 유지되는지 확인합니다.
      np.testing.assert_allclose(data[data_utils.DAY_PROGRESS], existing_day_progress_val)

  @parameterized.named_parameters(
      # 한글 주석: 필수 좌표(datetime, lon)가 누락된 경우 add_derived_vars 함수가 ValueError를 발생하는지 테스트합니다.
      dict(testcase_name="missing_datetime", coord_name="lon"), # datetime이 없고 lon만 있는 경우
      dict(testcase_name="missing_lon", coord_name="datetime"), # lon이 없고 datetime만 있는 경우
  )
  def test_add_derived_vars_missing_coordinate_raises_value_error(
      self, coord_name # 이 변수는 실제로는 누락된 좌표를 나타내는 데 사용되지 않음, 테스트 케이스 이름으로 구분
  ):
    missing_coord_test_name = self.id().split(".")[-1] # 현재 테스트 케이스 이름 가져오기
    if "missing_datetime" in missing_coord_test_name:
        # datetime 좌표가 없는 데이터셋 생성
        data = xa.Dataset(
            data_vars={"var1": (["x", "lon"], 8 * np.random.randn(2, 2))},
            coords={"lon": np.array([0.0, 0.5])},
        )
    elif "missing_lon" in missing_coord_test_name:
        # lon 좌표가 없는 데이터셋 생성
        data = xa.Dataset(
            data_vars={"var1": (["x", "datetime"], 8 * np.random.randn(2, 2))},
            coords={"datetime": np.array([datetime.datetime(2021,1,1)])},
        )
    else:
        raise ValueError("Invalid testcase name for coordinate check")

    with self.subTest(f"Missing {missing_coord_test_name.split('_')[-1]} coordinate"):
      with self.assertRaises(ValueError): # ValueError가 발생하는지 확인
        data_utils.add_derived_vars(data)

  def test_add_tisr_var_variable_added(self):
    # 한글 주석: add_tisr_var 함수가 TISR(대기 상층 입사 태양 복사량) 변수를 데이터셋에 올바르게 추가하는지 테스트합니다.
    data = xa.Dataset(
        data_vars={
            "var1": (["time", "lat", "lon"], np.full((2, 2, 2), 8.0))
        },
        coords={
            "lat": np.array([2.0, 1.0]),
            "lon": np.array([0.0, 0.5]),
            "time": np.array([100, 200], dtype="timedelta64[s]"),
            "datetime": xa.Variable( # datetime 좌표는 time 좌표와 연동되어야 함
                "time", np.array([datetime.datetime(2022,1,1,0,1,40), datetime.datetime(2022,1,1,0,3,20)], dtype="datetime64[ns]")
            ),
        },
    )

    data_utils.add_tisr_var(data) # TISR 변수 추가 함수 호출

    # 한글 주석: TISR 변수가 데이터셋에 추가되었는지 확인합니다.
    self.assertIn(data_utils.TISR, set(data.variables))

  def test_add_tisr_var_existing_var_not_overridden(self):
    # 한글 주석: add_tisr_var 함수가 이미 TISR 변수가 존재할 경우 덮어쓰지 않는지 테스트합니다.
    dims = ["time", "lat", "lon"]
    existing_tisr_val = 1200.0
    data = xa.Dataset(
        data_vars={
            "var1": (dims, np.full((2, 2, 2), 8.0)),
            data_utils.TISR: (dims, np.full((2, 2, 2), existing_tisr_val)), # 미리 TISR 변수 정의
        },
        coords={
            "lat": np.array([2.0, 1.0]),
            "lon": np.array([0.0, 0.5]),
            "time": np.array([100, 200], dtype="timedelta64[s]"),
            "datetime": xa.Variable(
                "time", np.array([datetime.datetime(2022,1,1,0,1,40), datetime.datetime(2022,1,1,0,3,20)], dtype="datetime64[ns]")
            ),
        },
    )

    data_utils.add_tisr_var(data) # TISR 변수 추가 함수 호출 (이미 변수 존재)

    # 한글 주석: TISR 변수가 기존 값으로 유지되는지 확인합니다.
    np.testing.assert_allclose(data[data_utils.TISR], existing_tisr_val)

  def test_add_tisr_var_works_with_batch_dim_size_one(self):
    # 한글 주석: add_tisr_var 함수가 배치 차원 크기가 1일 때 올바르게 작동하는지 테스트합니다.
    data = xa.Dataset(
        data_vars={
            "var1": (
                ["batch", "time", "lat", "lon"],
                np.full((1, 2, 2, 2), 8.0), # 배치 크기 1
            )
        },
        coords={
            "lat": np.array([2.0, 1.0]),
            "lon": np.array([0.0, 0.5]),
            "time": np.array([100, 200], dtype="timedelta64[s]"),
            "datetime": xa.Variable( # datetime도 배치 차원을 가져야 함
                ("batch", "time"), np.array([[datetime.datetime(2022,1,1,0,1,40), datetime.datetime(2022,1,1,0,3,20)]], dtype="datetime64[ns]")
            ),
        },
    )

    data_utils.add_tisr_var(data) # TISR 변수 추가 함수 호출

    # 한글 주석: TISR 변수가 데이터셋에 추가되었는지 확인합니다.
    self.assertIn(data_utils.TISR, set(data.variables))
    # 한글 주석: TISR 변수도 배치 차원을 가지는지 확인합니다.
    self.assertIn("batch", data[data_utils.TISR].dims)


  def test_add_tisr_var_fails_with_batch_dim_size_greater_than_one(self):
    # 한글 주석: add_tisr_var 함수가 배치 차원 크기가 1보다 클 때 ValueError를 발생하는지 테스트합니다.
    # 현재 solar_radiation.get_toa_incident_solar_radiation_for_xarray는 배치 차원을 직접 처리하지 않으므로,
    # add_tisr_var 내에서 squeeze('batch')를 시도하다가 배치 크기가 1이 아니면 오류가 발생합니다.
    data = xa.Dataset(
        data_vars={
            "var1": (
                ["batch", "time", "lat", "lon"],
                np.full((2, 2, 2, 2), 8.0), # 배치 크기 2
            )
        },
        coords={
            "lat": np.array([2.0, 1.0]),
            "lon": np.array([0.0, 0.5]),
            "time": np.array([100, 200], dtype="timedelta64[s]"),
            "datetime": xa.Variable(
                ("batch", "time"),
                np.array([[datetime.datetime(2022,1,1,0,1,40), datetime.datetime(2022,1,1,0,3,20)],
                          [datetime.datetime(2022,1,2,0,1,40), datetime.datetime(2022,1,2,0,3,20)]], dtype="datetime64[ns]"),
            ),
        },
    )

    with self.assertRaisesRegex(ValueError, r"cannot select a dimension"): # ValueError가 특정 메시지와 함께 발생하는지 확인
      data_utils.add_tisr_var(data)


if __name__ == "__main__":
  absltest.main() # 한글 주석: 테스트를 실행합니다.
