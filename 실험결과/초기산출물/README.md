# 초기 산출물

기존 루트의 efficientnet_b0/와 model.pt를 옮겼다. 정식 실험 번호와 정확한 대응이 확인되지 않아 001·002 결과로 임의 합산하지 않았다.

efficientnet_b0/runs.csv에는 workers=2·persistent_workers=True인 seed 42 완료 행이 있다. seed 43의 epoch 기록·가중치도 존재하지만 같은 실행의 완료 행은 없다. 유효한 workers별 속도 대조 실험으로 해석하지 않는다. model.pt의 학습 당시 노트북·실험 대응은 미확인이다. 가중치는 손실 없이 보존했다.

001의 정식 네 모델 비교 결과는 [001/model_comparison](../001_사전학습모델비교/model_comparison/)에 있다.
