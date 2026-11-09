import argparse
import csv
from pathlib import Path

import torch
from PIL import Image

from model import IMG_SIZE, EfficientNetEnsemble, build_valid_transform, combine_probabilities


def predict_probabilities(model, files):
	transform = build_valid_transform()
	rows = []
	with torch.inference_mode():
		for position in range(0, len(files), 32):
			images = []
			batch_files = files[position:position + 32]
			for path in batch_files:
				with Image.open(path) as image:
					images.append(transform(image.convert('RGB')))
			inputs = torch.stack(images)
			members = model.member_probabilities(inputs)
			flipped = model.member_probabilities(inputs.flip(dims=[3]))
			base_probs = combine_probabilities(members, 'T0.3').squeeze(1)
			tta_probs = combine_probabilities((members + flipped) / 2, 'T0.3').squeeze(1)
			blend_probs = (base_probs + tta_probs) / 2
			assert bool(torch.isfinite(blend_probs).all())
			for path, base_prob, tta_prob, blend_prob in zip(batch_files, base_probs, tta_probs, blend_probs):
				rows.append((path.stem, base_prob.item(), tta_prob.item(), blend_prob.item()))
	return rows


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--input', required=True)
	parser.add_argument('--output', required=True)
	parser.add_argument('--weights', default='model.pt')
	args = parser.parse_args()
	torch.set_num_threads(4)
	weights_path = Path(args.weights)
	if not weights_path.is_absolute():
		weights_path = Path(__file__).resolve().parent / weights_path
	checkpoint = torch.load(weights_path, map_location='cpu', weights_only=True)
	assert checkpoint['image_size'] == IMG_SIZE == 256
	model = EfficientNetEnsemble(checkpoint['state_dicts']).eval()
	n_params = sum(parameter.numel() for parameter in model.parameters())
	assert n_params <= 20_000_000
	files = sorted(path for path in Path(args.input).iterdir() if path.is_file() and (path.suffix.lower() in ('.jpg', '.jpeg', '.png')))
	assert files and (len({path.stem for path in files}) == len(files))
	rows = predict_probabilities(model, files)
	with Path(args.output).open('w', newline='', encoding='utf-8') as output:
		writer = csv.writer(output)
		writer.writerow(('id', 'prob'))
		writer.writerows((image_id, f'{blend_prob:.9f}') for image_id, _, _, blend_prob in rows)
	print(f'{len(rows)}장 / {n_params:,} parameters / CPU 기본·TTA 최종 확률 1:1 평균 / {args.output}')


if __name__ == '__main__':
	main()
