# 최종 제출 노트북과 추론

최종 제출: 실험 011 · 분할 seed 17 · 모델 seed 17/18/19/20 · 조건부 T0.3 TTA · Public Log Loss 0.0066.

| 노트북 | 역할 |
|---|---|
| [baseline_new.ipynb](baseline_new.ipynb) | 원래 셀 구성과 설명에 실제 seed 17 코드·설명을 반영한 제출 준비본. 모델·추론 코드 복원 기준 |
| [baseline_제출본.ipynb](baseline_제출본.ipynb) | 기존 학번·이름 폴더에 있던 baseline.ipynb를 이름만 바꿔 보존한 파일. 최종 ZIP 내부 코드와의 동일성을 이번 정리에서 새로 확정하지 않음 |
| [baseline.ipynb](baseline.ipynb) | 최종 폴더에 남아 있던 정리본 |
| [baseline_원본.ipynb](baseline_원본.ipynb) | 최종 폴더에 남아 있던 원본 |
| [학습 당시 baseline_test5.ipynb](../코드/baseline_test5.ipynb) | seed 17 학습 당시 실험 코드 |

노트북마다 설명·정리 상태가 달라 별도로 남겼다. 노트북의 모델·학습 코드는 변경하지 않았다. baseline_new의 과거 검증 기록에는 학습 함수·증강·분할·학습 루프 AST 동일성, 실제 B0 네 개 각 1 epoch 비교 시 모든 가중치 동일, 최종 가중치의 500장 예측 동일이 포함돼 있다. 이번 폴더 정리에서 전체 재학습은 하지 않았다.

가중치는 중복을 제거해 [../ensemble.pt](../ensemble.pt) 한 곳에 보관한다. 기존 최종 폴더 model.pt와 SHA-256이 같다. model.py와 predict.py는 baseline_new.ipynb의 %%writefile 셀에 들어 있어 중복 파일을 제거했다.

## 재학습 없이 추론

다음 코드를 **프로젝트 루트를 현재 디렉터리로 한 Python**에서 실행한다. 노트북의 파일 생성 셀 두 개만 추출하고 별도 임시 폴더에서 CPU 추론한다. 학습 셀은 실행하지 않는다.

```python
import json
import subprocess
import sys
import tempfile
from pathlib import Path

root = Path.cwd().resolve()
result = root / '실험결과' / '011_192크롭_패딩제거' / 'seed_17'
notebook = json.loads((result / '최종제출' / 'baseline_new.ipynb').read_text(encoding='utf-8'))
workspace = Path(tempfile.mkdtemp(prefix='aid-inference-'))
restored = set()
for cell in notebook['cells']:
	if cell['cell_type'] != 'code':
		continue
	source = ''.join(cell['source'])
	lines = source.splitlines(keepends=True)
	if not lines or not lines[0].startswith('%%writefile '):
		continue
	name = lines[0].strip().split()[1]
	if name in ('model.py', 'predict.py'):
		(workspace / name).write_text(''.join(lines[1:]), encoding='utf-8')
		restored.add(name)
assert restored == {'model.py', 'predict.py'}
output = workspace / 'pred.csv'
subprocess.run([sys.executable, '-B', str(workspace / 'predict.py'), '--weights', str(result / 'ensemble.pt'), '--input', str(root / '학습데이터' / 'test'), '--output', str(output)], cwd=workspace, check=True)
print(output)
```

기본 옵션은 T0.3 TTA다. 추론 시 사전학습 가중치를 다운로드하지 않는다. 입력은 최소 가로·세로 192픽셀인 이미지이며 결과는 id,prob 형식의 IT관 확률이다.

## 보존한 검증 근거

- 모델 파라미터: 16,035,316개.
- 실제 제출 CSV와 원본 CSV의 SHA-256: `0ec828ef0d94ff99d7b49e8ffc250713098463cb1f9ad31a0e4330dcb1004860`.
- 당시 학습 결과의 로컬 CPU 500장 TTA 재현: 약 18.55초, 확률 차이 0.
- 최종 준비 폴더의 다운로드 차단 CPU 재현: 약 19.92초, 제출 CSV와 바이트 단위 동일.
- 실제 Colab CPU 시간·사이트 최종 CSV 선택 상태는 미확인.

개별 예측 CSV·제출 ZIP·중복 TXT·JSON은 정리 요청에 따라 제거했다. 학습 설정과 가중치 출처는 [실험설정](../../실험설정.md), 상세 결과는 [실험결과](../../실험결과.md)에 남겼다.

## 2026-11-10 정리 후 검증

보존한 baseline_new.ipynb에서 모델·추론 코드만 추출해 CPU로 테스트 500장을 예측했다. 생성 CSV의 SHA-256이 위 실제 제출 CSV 해시와 정확히 일치했다. 학습·노트북 원문·가중치는 변경하지 않았다. 검증은 쓰기 가능한 별도 임시 작업 폴더에서 수행했다.
