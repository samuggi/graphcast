# Google DeepMind GraphCast 및 GenCast

이 패키지에는 연구 논문 [GraphCast](https://www.science.org/doi/10.1126/science.adi2336) 및 [GenCast](https://arxiv.org/abs/2312.15796)에 사용된 날씨 모델을 실행하고 학습시키기 위한 예제 코드가 포함되어 있습니다.

또한 [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/dm_graphcast)에서 사전 학습된 모델 가중치, 정규화 통계 및 예제 입력 데이터를 제공합니다.

전체 모델 학습에는 [ECMWF](https://www.ecmwf.int/)에서 제공하는 [ERA5](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5) 데이터셋 다운로드가 필요합니다. 이 데이터셋은 [Weatherbench2의 ERA5 데이터](https://weatherbench2.readthedocs.io/en/latest/data-guide.html#era5)에서 Zarr 형식으로 가장 잘 액세스할 수 있습니다.

운영 미세 조정을 위한 데이터는 유사하게 [Weatherbench2의 HRES 0번째 프레임 데이터](https://weatherbench2.readthedocs.io/en/latest/data-guide.html#ifs-hres-t-0-analysis)에서 액세스할 수 있습니다.

이러한 데이터셋은 별도의 이용 약관 또는 라이선스 조항의 적용을 받을 수 있습니다. 귀하의 해당 제3자 자료 사용은 해당 약관의 적용을 받으며, 사용하기 전에 적용 가능한 제한 사항이나 이용 약관을 준수할 수 있는지 확인해야 합니다.

## 모델 공통 파일 개요

*   `autoregressive.py`: JAX에서 미분 가능한 방식으로 각 단계에서 출력을 입력으로 자동 회귀적으로 다시 공급하여 예측 시퀀스를 생성하기 위해 (학습 및) 실행하는 데 사용되는 래퍼입니다.
*   `checkpoint.py`: 트리 직렬화 및 역직렬화를 위한 유틸리티입니다.
*   `data_utils.py`: 데이터 전처리를 위한 유틸리티입니다.
*   `deep_typed_graph_net.py`: 입력과 출력이 모두 각 노드 및 에지에 대한 특징의 플랫 벡터인 `TypedGraph`에서 작동하는 범용 심층 그래프 신경망(GNN)입니다.
*   `grid_mesh_connectivity.py`: 구 위의 일반 그리드와 삼각형 메쉬 간 변환 도구입니다.
*   `icosahedral_mesh.py`: 정이십면체 다중 메쉬 정의입니다.
*   `losses.py`: 위도 가중치를 포함한 손실 계산입니다.
*   `mlp.py`: 표준 조건 레이어가 있는 MLP 구축 유틸리티입니다.
*   `model_utils.py`: 입력 그리드 데이터에서 플랫 노드 및 에지 벡터 특징을 생성하고 노드 출력 벡터를 다중 레벨 그리드 데이터로 다시 조작하는 유틸리티입니다.
*   `normalization.py`: 과거 값에 따라 입력을 정규화하고 과거 시간 차이에 따라 대상을 정규화하는 데 사용되는 래퍼입니다.
*   `predictor_base.py`: 모델 및 모든 래퍼가 구현하는 예측기의 인터페이스를 정의합니다.
*   `rollout.py`: `autoregressive.py`와 유사하지만 더 길지만 미분 불가능한 궤적을 생성하기 위해 파이썬 루프를 사용하여 추론 시간에만 사용됩니다.
*   `typed_graph.py`: `TypedGraph` 정의입니다.
*   `typed_graph_net.py`: 더 깊은 모델을 구축하기 위해 결합할 수 있는 `TypedGraph`에 대해 정의된 간단한 그래프 신경망 빌딩 블록 구현입니다.
*   `xarray_jax.py`: JAX가 `xarray`와 함께 작동하도록 하는 래퍼입니다.
*   `xarray_tree.py`: `xarray`와 함께 작동하는 `tree.map_structure` 구현입니다.

## GenCast: 중기 날씨를 위한 확산 기반 앙상블 예측

이 패키지는 4개의 사전 학습된 모델을 제공합니다:

1.  `GenCast 0p25deg <2019`: 0.25도 해상도, 13개 기압 레벨 및 6회 미세 조정된 정이십면체 메쉬를 사용하는 GenCast 모델입니다. 이 모델은 1979년부터 2018년까지(포함)의 ERA5 데이터로 학습되었으며, 2019년 이후 연도에 대해 인과적으로 평가될 수 있습니다. 이 모델은 `GenCast: Diffusion-based ensemble forecasting for medium-range weather`(https://arxiv.org/abs/2312.15796) 논문에 설명되어 있습니다.

2.  `GenCast 0p25deg Operational <2022`: 0.25도 해상도, 13개 기압 레벨 및 6회 미세 조정된 정이십면체 메쉬를 사용하는 GenCast 모델입니다. 이 모델은 1979년부터 2018년까지의 ERA5 데이터로 학습되었고, 2016년부터 2021년까지의 HRES-fc0 데이터로 미세 조정되었으며, 2022년 이후 연도에 대해 인과적으로 평가될 수 있습니다. 이 모델은 운영 환경(즉, HRES-fc0에서 초기화됨)에서 예측을 수행할 수 있습니다.

3.  `GenCast 1p0deg <2019`: 1도 해상도, 13개 기압 레벨 및 5회 미세 조정된 정이십면체 메쉬를 사용하는 GenCast 모델입니다. 이 모델은 1979년부터 2018년까지의 ERA5 데이터로 학습되었으며, 2019년 이후 연도에 대해 인과적으로 평가될 수 있습니다. 이 모델은 0.25도 모델보다 메모리 사용량이 적습니다.

4. `GenCast 1p0deg Mini <2019`: 1도 해상도, 13개 기압 레벨 및 4회 미세 조정된 정이십면체 메쉬를 사용하는 GenCast 모델입니다. 이 모델은 1979년부터 2018년까지의 ERA5 데이터로 학습되었으며, 2019년 이후 연도에 대해 인과적으로 평가될 수 있습니다. 이 모델은 제공된 모델 중 메모리 사용량이 가장 작으며 저비용 데모(예: 무료 Colab 노트북에서 실행 가능)를 지원하기 위해 제공되었습니다. 성능은 합리적이지만 위의 GenCast 모델(1-3)의 성능을 대표하지는 않습니다. 참고로 ENS와의 성능을 비교한 스코어카드는 [docs/](https://github.com/google-deepmind/graphcast/blob/main/docs/GenCast_1p0deg_Mini_ENS_scorecard.png)에서 찾을 수 있습니다. 이 스코어카드에서 GenCast Mini는 ENS의 50개 멤버 앙상블과 달리 8개 멤버 앙상블만 사용하므로 공정한 비교를 위해 공정한 (편향되지 않은) CRPS를 사용합니다.

가장 좋은 시작점은 [Colaboratory](https://colab.research.google.com/github/deepmind/graphcast/blob/master/gencast_mini_demo.ipynb)에서 `gencast_mini_demo.ipynb`를 여는 것입니다. 이 노트북은 데이터 로드, 임의 가중치 생성 또는 `GenCast 1p0deg Mini <2019` 스냅샷 로드, 예측 생성, 손실 계산 및 기울기 계산의 예를 보여줍니다. GenCast 아키텍처의 단일 단계 구현은 `gencast.py`에 제공되며 관련 데이터, 가중치 및 통계는 Google Cloud Bucket의 `gencast/` 하위 디렉토리에 있습니다.

### Google Cloud 컴퓨팅에서 GenCast 실행 지침

[cloud_vm_setup.md](https://github.com/google-deepmind/graphcast/blob/main/docs/cloud_vm_setup.md)에는 Google Cloud TPU VM 시작에 대한 자세한 지침이 포함되어 있습니다. 이를 통해 [Colaboratory](https://colab.research.google.com/github/deepmind/graphcast/blob/master/gencast_demo_cloud_vm.ipynb)를 통해 별도의 `gencast_demo_cloud_vm.ipynb`에서 모델(1-3)을 실행할 수 있습니다.

이 문서는 또한 GPU에서 GenCast를 실행하기 위한 [지침](https://github.com/google-deepmind/graphcast/blob/main/docs/cloud_vm_setup.md#running-inference-on-gpu)을 제공합니다. 이를 위해서는 다른 어텐션 구현을 사용해야 합니다.

### 관련 라이브러리 파일에 대한 간략한 설명

*   `denoiser.py`: 단일 단계 예측을 위한 GenCast 디노이저입니다.
*   `denoisers_base.py`: 디노이저의 인터페이스를 정의합니다.
*   `dpm_solver_plus_plus_2s.py`: [1]의 DPM-Solver++ 2S를 사용하는 샘플러입니다.
*   `gencast.py`: 디노이저로 래핑된 GenCast 모델 아키텍처를 샘플러와 결합하여 예측을 생성합니다.
*   `nan_cleaning.py`: NaN이 제거된 데이터로 작업할 수 있도록 예측기를 래핑합니다. 해수면 온도의 NaN을 제거하는 데 사용됩니다.
*   `samplers_base.py`: 샘플러의 인터페이스를 정의합니다.
*   `samplers_utils.py`: 샘플러를 위한 유틸리티 메서드입니다.
*   `sparse_transformer.py`: 입력과 출력이 모두 각 노드 및 에지에 대한 특징의 플랫 벡터인 `TypedGraph`에서 작동하는 범용 희소 트랜스포머입니다. `predictor.py`는 메쉬 GNN에 대해 이 중 하나를 사용합니다.
*   `sparse_transformer_utils.py`: 희소 트랜스포머를 위한 유틸리티 메서드입니다.
*   `transformer.py`: 입력 그래프의 노드에서 선행 두 축을 바꾸는 메쉬 트랜스포머를 래핑합니다.

[1] DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models, https://arxiv.org/abs/2211.01095

## GraphCast: 숙련된 중기 전지구 날씨 예측 학습

이 패키지는 세 가지 사전 학습된 모델을 제공합니다:

1.  `GraphCast`: GraphCast 논문에 사용된 고해상도 모델(0.25도 해상도, 37개 기압 레벨)로, 1979년부터 2017년까지의 ERA5 데이터로 학습되었습니다.

2.  `GraphCast_small`: GraphCast의 더 작고 저해상도 버전(1도 해상도, 13개 기압 레벨, 더 작은 메쉬)으로, 1979년부터 2015년까지의 ERA5 데이터로 학습되었으며, 메모리 및 컴퓨팅 제약이 낮은 모델을 실행하는 데 유용합니다.

3.  `GraphCast_operational`: 1979년부터 2017년까지의 ERA5 데이터로 사전 학습되고 2016년부터 2021년까지의 HRES 데이터로 미세 조정된 고해상도 모델(0.25도 해상도, 13개 기압 레벨)입니다. 이 모델은 HRES 데이터에서 초기화할 수 있습니다(강수량 입력이 필요하지 않음).

가장 좋은 시작점은 [Colaboratory](https://colab.research.google.com/github/deepmind/graphcast/blob/master/graphcast_demo.ipynb)에서 `graphcast_demo.ipynb`를 여는 것입니다. 이 노트북은 데이터 로드, 임의 가중치 생성 또는 사전 학습된 스냅샷 로드, 예측 생성, 손실 계산 및 기울기 계산의 예를 보여줍니다. GraphCast 아키텍처의 단일 단계 구현은 `graphcast.py`에 제공되며 관련 데이터, 가중치 및 통계는 Google Cloud Bucket의 `graphcast/` 하위 디렉토리에 있습니다.

경고: 이전 버전과의 호환성을 위해 GraphCast 데이터를 버킷의 최상위 수준에도 남겨두었습니다. 이들은 결국 `graphcast/` 하위 디렉토리를 위해 삭제될 것입니다.

### 관련 라이브러리 파일에 대한 간략한 설명:

*   `casting.py`: BFloat16 정밀도를 사용하여 GraphCast가 작동하도록 하는 데 사용되는 래퍼입니다.
*   `graphcast.py`: 단일 단계 예측을 위한 주요 GraphCast 모델 아키텍처입니다.
*   `solar_radiation.py`: ERA5와 호환되는 대기 상단(TOA) 입사 태양 복사를 계산합니다. 이는 강제 변수로 사용되므로 운영 환경에서 목표 리드 타임에 대해 계산해야 합니다.

## 종속성

[Chex](https://github.com/deepmind/chex),
[Dask](https://github.com/dask/dask),
[Dinosaur](https://github.com/google-research/dinosaur),
[Haiku](https://github.com/deepmind/dm-haiku),
[JAX](https://github.com/google/jax),
[JAXline](https://github.com/deepmind/jaxline),
[Jraph](https://github.com/deepmind/jraph),
[Numpy](https://numpy.org/),
[Pandas](https://pandas.pydata.org/),
[Python](https://www.python.org/),
[SciPy](https://scipy.org/),
[Tree](https://github.com/deepmind/tree),
[Trimesh](https://github.com/mikedh/trimesh),
[XArray](https://github.com/pydata/xarray) 및
[XArray-TensorStore](https://github.com/google/xarray-tensorstore).


## 라이선스 및 면책 조항

Colab 노트북 및 관련 코드는 Apache License, Version 2.0에 따라 라이선스가 부여됩니다. 라이선스 사본은 다음에서 얻을 수 있습니다: https://www.apache.org/licenses/LICENSE-2.0.

모델 가중치는 Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) 조건에 따라 사용할 수 있도록 제공됩니다. 라이선스 사본은 다음에서 얻을 수 있습니다: https://creativecommons.org/licenses/by-nc-sa/4.0/.

이것은 공식적으로 지원되는 Google 제품이 아닙니다.

관련 법률에서 요구하거나 서면으로 동의하지 않는 한, Apache 2.0 또는 CC-BY-NC-SA 4.0 라이선스에 따라 여기에 배포된 모든 소프트웨어 및 자료는 명시적이든 묵시적이든 어떠한 종류의 보증이나 조건 없이 "있는 그대로" 배포됩니다. 해당 라이선스에 따른 특정 언어 관리 권한 및 제한 사항은 라이선스를 참조하십시오.

GenCast 및 GraphCast는 실험적인 연구 프로젝트의 일부입니다. GenCast, GraphCast 또는 생성된 모든 결과물의 사용 또는 배포의 적절성을 결정하는 것은 전적으로 귀하의 책임이며, GenCast, GraphCast 및 결과물의 사용 또는 배포와 관련 라이선스에 따라 Google이 귀하에게 부여한 권리 및 권한 행사에 따른 모든 위험을 감수해야 합니다. GenCast, GraphCast 또는 생성된 모든 결과물을 신뢰, 게시, 다운로드 또는 기타 방식으로 사용하기 전에 신중하게 판단하십시오. GenCast, GraphCast 또는 생성된 모든 결과물은 (i) 정부 기상 기관 또는 부서에서 발표한 데이터를 기반으로 하지 않으며, (ii) 해당 기관과의 협력을 통해 생성되지 않았으며, (iii) 해당 기관의 승인을 받지 않았으며 어떠한 방식으로든 해당 기관에서 발표한 공식 경보, 경고 또는 통지를 대체하지 않습니다.

Copyright 2024 DeepMind Technologies Limited.


## 인용

이 작업을 사용하는 경우 다음 논문을 인용하는 것을 고려하십시오 ([블로그 게시물](https://deepmind.google/discover/blog/graphcast-ai-model-for-faster-and-more-accurate-global-weather-forecasting/), [Science](https://www.science.org/doi/10.1126/science.adi2336), [arXiv](https://arxiv.org/abs/2212.12794), [arxiv GenCast](https://arxiv.org/abs/2312.15796)):

```latex
@article{lam2023learning,
  title={Learning skillful medium-range global weather forecasting},
  author={Lam, Remi and Sanchez-Gonzalez, Alvaro and Willson, Matthew and Wirnsberger, Peter and Fortunato, Meire and Alet, Ferran and Ravuri, Suman and Ewalds, Timo and Eaton-Rosen, Zach and Hu, Weihua and others},
  journal={Science},
  volume={382},
  number={6677},
  pages={1416--1421},
  year={2023},
  publisher={American Association for the Advancement of Science}
}
```


```latex
@article{price2023gencast,
  title={GenCast: Diffusion-based ensemble forecasting for medium-range weather},
  author={Price, Ilan and Sanchez-Gonzalez, Alvaro and Alet, Ferran and Andersson, Tom R and El-Kadi, Andrew and Masters, Dominic and Ewalds, Timo and Stott, Jacklynn and Mohamed, Shakir and Battaglia, Peter and Lam, Remi and Willson, Matthew},
  journal={arXiv preprint arXiv:2312.15796},
  year={2023}
}
```

## 감사의 말

(i) GenCast 및 GraphCast는 다음의 별도 라이브러리 및 패키지와 통신하거나 참조하며, colab 노트북에는 모델의 입력으로 사용할 수 있는 ECMWF의 ERA5 및 HRES 데이터의 몇 가지 예가 포함되어 있습니다.
Google에서 수정한 유럽 중기 예보 센터(ECMWF)의 데이터 및 제품.
수정된 코페르니쿠스 기후 변화 서비스 정보 2023. 유럽 위원회나 ECMWF는 포함된 코페르니쿠스 정보 또는 데이터의 사용에 대해 책임을 지지 않습니다.
ECMWF HRES 데이터셋
저작권 표시: 저작권 "© 2023 European Centre for Medium-Range Weather Forecasts (ECMWF)".
출처: www.ecmwf.int
라이선스 설명: ECMWF 공개 데이터는 Creative Commons Attribution 4.0 International (CC BY 4.0)에 따라 게시됩니다. https://creativecommons.org/licenses/by/4.0/
면책 조항: ECMWF는 데이터의 오류나 누락, 가용성 또는 사용으로 인해 발생하는 모든 손실이나 손해에 대해 어떠한 책임도 지지 않습니다.

위에 언급된 제3자 자료의 사용은 별도의 이용 약관 또는 라이선스 조항의 적용을 받을 수 있습니다. 귀하의 제3자 자료 사용은 해당 약관의 적용을 받으며, 사용하기 전에 적용 가능한 제한 사항이나 이용 약관을 준수할 수 있는지 확인해야 합니다.


## 연락처

피드백 및 질문은 gencast@google.com으로 문의하십시오.
