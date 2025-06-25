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
"""Check that the checkpoint serialization is reversable."""
# 한글 주석: 체크포인트 직렬화가 가역적인지 (저장했다가 다시 로드했을 때 동일한지) 확인하는 테스트 모듈입니다.

import dataclasses
import io
from typing import Any, Optional, Union # 한글 주석: 타입 힌트를 위한 모듈 임포트

from absl.testing import absltest # 한글 주석: Absl 테스트 프레임워크 임포트
from graphcast import checkpoint # 한글 주석: 테스트 대상인 checkpoint 모듈 임포트
import numpy as np


@dataclasses.dataclass
class SubConfig:
  # 한글 주석: 테스트에 사용될 간단한 데이터 클래스 정의 (Config 내부에 중첩됨)
  a: int
  b: str


@dataclasses.dataclass
class Config:
  # 한글 주석: 다양한 데이터 타입을 포함하는 테스트용 설정 데이터 클래스 정의
  bt: bool
  bf: bool
  i: int
  f: float
  o1: Optional[int] # 한글 주석: Optional 타입 (값이 있거나 None)
  o2: Optional[int]
  o3: Union[int, None] # 한글 주석: Union 타입 (int 또는 None)
  o4: Union[int, None]
  o5: int | None # 한글 주석: Python 3.10+의 새로운 Union 표현 방식
  o6: int | None
  li: list[int] # 한글 주석: 정수 리스트
  ls: list[str] # 한글 주석: 문자열 리스트
  ldc: list[SubConfig] # 한글 주석: SubConfig 객체의 리스트
  tf: tuple[float, ...] # 한글 주석: 임의 길이의 float 튜플
  ts: tuple[str, ...] # 한글 주석: 임의 길이의 str 튜플
  t: tuple[str, int, SubConfig] # 한글 주석: 고정 길이 및 다양한 타입의 튜플
  tdc: tuple[SubConfig, ...] # 한글 주석: 임의 길이의 SubConfig 튜플
  dsi: dict[str, int] # 한글 주석: 문자열 키, 정수 값의 딕셔너리
  dss: dict[str, str] # 한글 주석: 문자열 키, 문자열 값의 딕셔너리
  dis: dict[int, str] # 한글 주석: 정수 키, 문자열 값의 딕셔너리
  dsdis: dict[str, dict[int, str]] # 한글 주석: 중첩된 딕셔너리
  dc: SubConfig # 한글 주석: SubConfig 객체
  dco: Optional[SubConfig] # 한글 주석: Optional SubConfig 객체
  ddc: dict[str, SubConfig] # 한글 주석: 문자열 키, SubConfig 값의 딕셔너리


@dataclasses.dataclass
class Checkpoint:
  # 한글 주석: 실제 체크포인트 구조를 모방한 테스트용 데이터 클래스
  params: dict[str, Any] # 한글 주석: 모델 파라미터 (다양한 중첩 구조 가능)
  config: Config # 한글 주석: 위에서 정의한 Config 객체


class DataclassTest(absltest.TestCase):
  # 한글 주석: 데이터 클래스 직렬화 및 역직렬화 테스트를 위한 클래스

  def test_serialize_dataclass(self):
    # 한글 주석: 데이터 클래스의 직렬화 및 역직렬화 과정을 테스트합니다.
    # 복잡한 중첩 구조와 다양한 데이터 타입을 가진 Checkpoint 객체를 생성하고,
    # 이를 파일(메모리 버퍼)에 저장했다가 다시 로드하여 원본과 동일한지 확인합니다.

    # 테스트용 Checkpoint 객체 생성
    ckpt = Checkpoint(
        params={
            "layer1": {
                "w": np.arange(10).reshape(2, 5), # 넘파이 배열
                "b": np.array([2, 6]),
            },
            "layer2": {
                "w": np.arange(8).reshape(2, 4),
                "b": np.array([2, 6]),
            },
            "blah": np.array([3, 9]),
        },
        config=Config( # 위에서 정의한 Config 클래스의 인스턴스
            bt=True,
            bf=False,
            i=42,
            f=3.14,
            o1=1,
            o2=None, # Optional 타입에 None 할당
            o3=2,
            o4=None,
            o5=3,
            o6=None,
            li=[12, 9, 7, 15, 16, 14, 1, 6, 11, 4, 10, 5, 13, 3, 8, 2],
            ls=list("qhjfdxtpzgemryoikwvblcaus"),
            ldc=[SubConfig(1, "hello"), SubConfig(2, "world")],
            tf=(1, 4, 2, 10, 5, 9, 13, 16, 15, 8, 12, 7, 11, 14, 3, 6),
            ts=("hello", "world"),
            t=("foo", 42, SubConfig(1, "bar")),
            tdc=(SubConfig(1, "hello"), SubConfig(2, "world")),
            dsi={"a": 1, "b": 2, "c": 3},
            dss={"d": "e", "f": "g"},
            dis={1: "a", 2: "b", 3: "c"},
            dsdis={"a": {1: "hello", 2: "world"}, "b": {1: "world"}},
            dc=SubConfig(1, "hello"),
            dco=None, # Optional 데이터 클래스에 None 할당
            ddc={"a": SubConfig(1, "hello"), "b": SubConfig(2, "world")},
        ))

    buffer = io.BytesIO() # 한글 주석: 메모리 내 바이너리 스트림을 사용합니다.
    checkpoint.dump(buffer, ckpt) # 한글 주석: Checkpoint 객체를 버퍼에 덤프합니다.
    buffer.seek(0) # 한글 주석: 버퍼의 읽기 위치를 처음으로 되돌립니다.
    ckpt2 = checkpoint.load(buffer, Checkpoint) # 한글 주석: 버퍼에서 Checkpoint 객체를 로드합니다.

    # 한글 주석: 로드된 객체의 파라미터들이 원본과 동일한지 넘파이 배열 단위로 비교합니다.
    np.testing.assert_array_equal(ckpt.params["layer1"]["w"],
                                  ckpt2.params["layer1"]["w"])
    np.testing.assert_array_equal(ckpt.params["layer1"]["b"],
                                  ckpt2.params["layer1"]["b"])
    np.testing.assert_array_equal(ckpt.params["layer2"]["w"],
                                  ckpt2.params["layer2"]["w"])
    np.testing.assert_array_equal(ckpt.params["layer2"]["b"],
                                  ckpt2.params["layer2"]["b"])
    np.testing.assert_array_equal(ckpt.params["blah"], ckpt2.params["blah"])
    # 한글 주석: 로드된 객체의 config 부분이 원본과 동일한지 비교합니다.
    # 데이터 클래스는 __eq__ 메서드가 자동으로 생성되므로 직접 비교 가능합니다.
    self.assertEqual(ckpt.config, ckpt2.config)


if __name__ == "__main__":
  absltest.main() # 한글 주석: 테스트를 실행합니다.
