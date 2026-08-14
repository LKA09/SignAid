# SignAid submission

## 실행

Python 3.11+와 Node.js 20.19+ 환경에서 다음 명령을 실행합니다.

```powershell
python -m pip install -e ".[dev]"
signaid
```

제출 ZIP에는 빌드된 웹 화면도 포함되어 있습니다. `http://127.0.0.1:8000`에서 실행되며 종료는 `Ctrl+C`입니다.

## 포함 데이터

- AI Hub 3D 신체·양손·얼굴 랜드마크에서 선택 추출한 21개 NPZ
- AI Hub 2D 키포인트 기반 표현 1개
- 파생 모션 전체 용량 약 0.5MB
- 응급 문장 검색 인덱스와 로컬 VRM 아바타

원본 AI Hub ZIP, 임시 추출 JSON/XML, 영상, 체크포인트, `node_modules`는 포함하지 않습니다. 파생 데이터의 제출·재배포 가능 범위는 대회 규정과 AI Hub 이용약관을 별도로 확인해야 합니다.

모션은 AI Hub 주석 구간에 정렬됐지만 한국수어 전문가 검수 전입니다. 앱 역시 이를 명시하며, 미지원 표현은 시연용 동작으로 구분합니다.

## 제출 ZIP 다시 만들기

```powershell
python scripts\build_submission.py
```
