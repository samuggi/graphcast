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
"""Serialize and deserialize trees."""
# 한글 주석: 트리 구조의 데이터를 직렬화(serialize)하고 역직렬화(deserialize)하는 모듈입니다.

import dataclasses
import io
import types
from typing import Any, BinaryIO, Optional, TypeVar # 한글 주석: 타입 힌트를 위한 모듈 임포트

import numpy as np

_T = TypeVar("_T") # 한글 주석: 제네릭 타입을 위한 타입 변수 정의


def dump(dest: BinaryIO, value: Any) -> None:
  """Dump a tree of dicts/dataclasses to a file object.
  # 한글 주석: 딕셔너리/데이터클래스의 트리 구조를 파일 객체에 덤프(저장)합니다.

  Args:
  # 한글 주석: 인수
    dest: a file object to write to.
    # 한글 주석: dest: 데이터를 쓸 파일 객체입니다.
    value: A tree of dicts, lists, tuples and dataclasses of numpy arrays and
      other basic types. Unions are not supported, other than Optional/None
      which is only supported in dataclasses, not in dicts, lists or tuples.
      All leaves must be coercible to a numpy array, and recoverable as a single
      arg to a type.
    # 한글 주석: value: 넘파이 배열 및 기타 기본 타입으로 구성된 딕셔너리, 리스트, 튜플, 데이터클래스의 트리 구조입니다.
    # Optional/None을 제외한 Union 타입은 지원되지 않으며, Optional/None은 데이터클래스에서만 지원됩니다 (딕셔너리, 리스트, 튜플에서는 지원 안 됨).
    # 모든 리프(leaf) 노드는 넘파이 배열로 변환 가능해야 하며, 단일 인수로 타입으로 복구 가능해야 합니다.
  """
  buffer = io.BytesIO()  # In case the destination doesn't support seeking. # 한글 주석: 대상 파일 객체가 탐색(seeking)을 지원하지 않는 경우를 대비한 버퍼입니다.
  np.savez(buffer, **_flatten(value)) # 한글 주석: 값을 평탄화하여 버퍼에 .npz 형식으로 저장합니다.
  dest.write(buffer.getvalue()) # 한글 주석: 버퍼의 내용을 대상 파일 객체에 씁니다.


def load(source: BinaryIO, typ: type[_T]) -> _T:
  """Load from a file object and convert it to the specified type.
  # 한글 주석: 파일 객체에서 데이터를 로드하고 지정된 타입으로 변환합니다.

  Args:
  # 한글 주석: 인수
    source: a file object to read from.
    # 한글 주석: source: 데이터를 읽을 파일 객체입니다.
    typ: a type object that acts as a schema for deserialization. It must match
      what was serialized. If a type is Any, it will be returned however numpy
      serialized it, which is what you want for a tree of numpy arrays.
    # 한글 주석: typ: 역직렬화를 위한 스키마 역할을 하는 타입 객체입니다. 직렬화된 것과 일치해야 합니다.
    # 타입이 Any인 경우, 넘파이가 직렬화한 방식 그대로 반환됩니다 (넘파이 배열 트리인 경우 원하는 동작).

  Returns:
  # 한글 주석: 반환값
    the deserialized value as the specified type.
    # 한글 주석: 역직렬화된 값을 지정된 타입으로 반환합니다.
  """
  return _convert_types(typ, _unflatten(np.load(source))) # 한글 주석: .npz 파일에서 로드하고, 평탄화 해제한 후 타입을 변환하여 반환합니다.


_SEP = ":" # 한글 주석: 평탄화된 딕셔너리의 키를 구분하는 데 사용되는 구분자입니다.


def _flatten(tree: Any) -> dict[str, Any]:
  """Flatten a tree of dicts/dataclasses/lists/tuples to a single dict."""
  # 한글 주석: 딕셔너리/데이터클래스/리스트/튜플의 트리 구조를 단일 딕셔너리로 평탄화합니다.
  if dataclasses.is_dataclass(tree):
    # Don't use dataclasses.asdict as it is recursive so skips dropping None.
    # 한글 주석: dataclasses.asdict는 재귀적이며 None을 생략하는 것을 건너뛰므로 사용하지 않습니다.
    # 데이터클래스의 경우, None이 아닌 필드만 포함하는 딕셔너리로 변환합니다.
    tree = {f.name: v for f in dataclasses.fields(tree)
            if (v := getattr(tree, f.name)) is not None}
  elif isinstance(tree, (list, tuple)):
    # 리스트나 튜플인 경우, 인덱스를 키로 하는 딕셔너리로 변환합니다.
    tree = dict(enumerate(tree))

  assert isinstance(tree, dict) # 한글 주석: 변환 후에는 반드시 딕셔너리 타입이어야 합니다.

  flat = {} # 한글 주석: 평탄화된 결과를 저장할 딕셔너리입니다.
  for k, v in tree.items():
    k = str(k) # 한글 주석: 키를 문자열로 변환합니다.
    assert _SEP not in k # 한글 주석: 키에 구분자가 포함되어 있지 않은지 확인합니다.
    if dataclasses.is_dataclass(v) or isinstance(v, (dict, list, tuple)):
      # 한글 주석: 값이 데이터클래스, 딕셔너리, 리스트, 튜플인 경우 재귀적으로 평탄화합니다.
      for a, b in _flatten(v).items():
        flat[f"{k}{_SEP}{a}"] = b # 한글 주석: 부모 키와 현재 키를 구분자로 연결하여 새로운 키를 만듭니다.
    else:
      # 한글 주석: 값이 리프 노드인 경우
      assert v is not None # 한글 주석: 값은 None이 아니어야 합니다 (데이터클래스가 아닌 경우).
      flat[k] = v # 한글 주석: 그대로 저장합니다.
  return flat


def _unflatten(flat: dict[str, Any]) -> dict[str, Any]:
  """Unflatten a dict to a tree of dicts."""
  # 한글 주석: 단일 딕셔너리를 딕셔너리의 트리 구조로 평탄화 해제합니다.
  tree = {} # 한글 주석: 평탄화 해제된 결과를 저장할 딕셔너리입니다.
  for flat_key, v in flat.items():
    node = tree
    keys = flat_key.split(_SEP) # 한글 주석: 구분자를 기준으로 키를 분리합니다.
    for k in keys[:-1]: # 한글 주석: 마지막 키를 제외한 각 키에 대해
      if k not in node:
        node[k] = {} # 한글 주석: 경로가 없으면 새로 생성합니다.
      node = node[k] # 한글 주석: 다음 노드로 이동합니다.
    node[keys[-1]] = v # 한글 주석: 마지막 키에 값을 할당합니다.
  return tree


def _convert_types(typ: type[_T], value: Any) -> _T:
  """Convert some structure into the given type. The structures must match."""
  # 한글 주석: 어떤 구조를 주어진 타입으로 변환합니다. 구조는 반드시 일치해야 합니다.
  if typ in (Any, ...): # 한글 주석: 타입이 Any 또는 Ellipsis(...)이면 값을 그대로 반환합니다.
    return value

  if typ in (int, float, str, bool): # 한글 주석: 기본 타입인 경우 해당 타입으로 변환합니다.
    return typ(value)

  if typ is np.ndarray: # 한글 주석: 타입이 넘파이 배열인 경우
    assert isinstance(value, np.ndarray) # 한글 주석: 값이 넘파이 배열인지 확인합니다.
    return value # 한글 주석: 그대로 반환합니다.

  if dataclasses.is_dataclass(typ): # 한글 주석: 타입이 데이터클래스인 경우
    kwargs = {}
    for f in dataclasses.fields(typ): # 한글 주석: 데이터클래스의 각 필드에 대해
      # Only support Optional for dataclasses, as numpy can't serialize it
      # directly (without pickle), and dataclasses are the only case where we
      # can know the full set of values and types and therefore know the
      # non-existence must mean None.
      # 한글 주석: 넘파이는 Optional을 직접 직렬화할 수 없고 (pickle 제외),
      # 데이터클래스는 전체 값과 타입을 알 수 있어 존재하지 않음이 None을 의미함을 알 수 있는 유일한 경우이므로
      # 데이터클래스에 대해서만 Optional을 지원합니다.
      if isinstance(f.type, (types.UnionType, type(Optional[int]))): # 필드 타입이 Union 또는 Optional인 경우
        constructors = [t for t in f.type.__args__ if t is not types.NoneType] # None이 아닌 타입 인자들을 가져옵니다.
        if len(constructors) != 1:
          raise TypeError(
              "Optional은 작동하지만, None을 제외한 다른 것과의 Union은 작동하지 않습니다.")
        if f.name not in value: # 필드 이름이 값에 없으면
          kwargs[f.name] = None # None으로 설정합니다.
          continue
        constructor = constructors[0] # 실제 생성자 타입을 가져옵니다.
      else:
        constructor = f.type # 필드 타입을 생성자로 사용합니다.

      if f.name in value: # 필드 이름이 값에 있으면
        kwargs[f.name] = _convert_types(constructor, value[f.name]) # 재귀적으로 타입을 변환하여 할당합니다.
      else:
        raise ValueError(f"누락된 값: {f.name}")
    return typ(**kwargs) # 변환된 값들로 데이터클래스 객체를 생성하여 반환합니다.

  base_type = getattr(typ, "__origin__", None) # 한글 주석: 제네릭 타입의 원형(origin)을 가져옵니다 (예: list[int]의 list).

  if base_type is dict: # 한글 주석: 원형이 dict인 경우
    assert len(typ.__args__) == 2 # 한글 주석: 타입 인자가 2개인지 확인합니다 (키 타입, 값 타입).
    key_type, value_type = typ.__args__
    return {_convert_types(key_type, k): _convert_types(value_type, v) # 키와 값을 재귀적으로 변환합니다.
            for k, v in value.items()}

  if base_type is list: # 한글 주석: 원형이 list인 경우
    assert len(typ.__args__) == 1 # 한글 주석: 타입 인자가 1개인지 확인합니다 (값 타입).
    value_type = typ.__args__[0]
    # 저장 시 enumerate로 인덱스를 키로 사용했으므로, 정렬하여 원래 순서대로 리스트를 만듭니다.
    return [_convert_types(value_type, v)
            for _, v in sorted(value.items(), key=lambda x: int(x[0]))]

  if base_type is tuple: # 한글 주석: 원형이 tuple인 경우
    if len(typ.__args__) == 2 and typ.__args__[1] == ...:
      # An arbitrary length tuple of a single type, eg: tuple[int, ...]
      # 한글 주석: 단일 타입의 임의 길이 튜플인 경우 (예: tuple[int, ...])
      value_type = typ.__args__[0]
      return tuple(_convert_types(value_type, v)
                   for _, v in sorted(value.items(), key=lambda x: int(x[0])))
    else:
      # A fixed length tuple of arbitrary types, eg: tuple[int, str, float]
      # 한글 주석: 임의 타입의 고정 길이 튜플인 경우 (예: tuple[int, str, float])
      assert len(typ.__args__) == len(value) # 타입 인자 수와 값의 길이가 같은지 확인합니다.
      return tuple(
          _convert_types(t, v)
          for t, (_, v) in zip(
              typ.__args__, sorted(value.items(), key=lambda x: int(x[0]))))

  # This is probably unreachable with reasonable serializable inputs.
  # 한글 주석: 합리적인 직렬화 가능한 입력으로는 아마 도달할 수 없는 부분입니다.
  try:
    return typ(value) # 주어진 타입으로 직접 변환 시도
  except TypeError as e:
    raise TypeError(
        "_convert_types는 타입 인자가 유효한 생성자인 타입 (예: tuple은 괜찮지만 Tuple은 아님)으로 정의된 데이터클래스이고, "
        "넘파이 배열을 유일한 인수로 받아들일 것으로 예상합니다.") from e
