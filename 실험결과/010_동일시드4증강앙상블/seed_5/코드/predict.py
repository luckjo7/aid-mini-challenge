import argparse
import glob
import os

import torch
from PIL import Image
from model import IMG_SIZE, TEMPERATURE_RULES, EfficientNetEnsemble, build_valid_transform, combine_probabilities

PARAM_LIMIT = 20_000_000


def load_images(input_dir):
	exts = (".jpg", ".jpeg", ".png")
	return sorted(f for f in glob.glob(os.path.join(input_dir, "*")) if f.lower().endswith(exts))


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--input", required=True, help="테스트 이미지 폴더")
	parser.add_argument("--output", required=True, help="출력 CSV 경로")
	parser.add_argument("--weights", default="model.pt", help="앙상블 가중치 파일")
	parser.add_argument("--temperature-rule", choices=TEMPERATURE_RULES, default="T0.3", help="T0.3은 기존 조건부 규칙, T0.5는 전체 0.5 규칙")
	parser.add_argument("--tta", action="store_true", help="원본·좌우 반전 확률을 평균한 뒤 조건부 T0.3 적용")
	args = parser.parse_args()
	if args.tta and (args.temperature_rule != "T0.3"):
		parser.error("--tta는 --temperature-rule T0.3과 함께 사용하세요.")

	device = torch.device("cpu")
	weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.weights)
	checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
	assert checkpoint["image_size"] == IMG_SIZE, "가중치의 입력 크기를 확인하세요."
	model = EfficientNetEnsemble(checkpoint["state_dicts"]).to(device)
	model.eval()

	n_params = sum(p.numel() for p in model.parameters())
	print(f"앙상블 파라미터 수: {n_params:,} (제한 {PARAM_LIMIT:,})")
	assert n_params <= PARAM_LIMIT, "파라미터 20M 초과!"

	transform = build_valid_transform()
	files = load_images(args.input)
	assert len(files) > 0, f"{args.input}에서 이미지를 찾지 못했습니다."

	rows = ["id,prob"]
	with torch.inference_mode():
		for i in range(0, len(files), 32):
			batch_files = files[i:i + 32]
			images = []
			for file in batch_files:
				with Image.open(file) as image:
					images.append(transform(image.convert("RGB")))
			inputs = torch.stack(images).to(device)
			if args.tta:
				member_probs = model.member_probabilities(inputs)
				flipped_member_probs = model.member_probabilities(inputs.flip(dims=[3]))
				probs = combine_probabilities((member_probs + flipped_member_probs) / 2, "T0.3").squeeze(1)
			else:
				probs = model(inputs, temperature_rule=args.temperature_rule).squeeze(1)
			for file, prob in zip(batch_files, probs):
				rows.append(f"{os.path.splitext(os.path.basename(file))[0]},{prob.item():.6f}")

	with open(args.output, "w", encoding="utf-8") as f:
		f.write("\n".join(rows) + "\n")
	print(f"{args.output} 저장 완료 ({len(files)}장)")


if __name__ == "__main__":
	main()
