import argparse
import image_classifier_utils as icu
import sys
import json

predict_parser = argparse.ArgumentParser(description='Predicts the flower name from an image \
                                                      along with probability of that name.')

predict_parser.add_argument('input', nargs=1, action="store", type=str, default="./flowers/test/1/image_06743.jpg",
                          help='path of image file')

predict_parser.add_argument('checkpoint', nargs=1, action="store", type=str, default="./checkpoints/checkpoint.pth",
                          help='path of the models checkpoint')

predict_parser.add_argument('--category_to_names', dest='category_to_names', action="store", default='cat_to_name.json',
                          help='path to mapping file for category to names')

predict_parser.add_argument('--top_k', dest='top_k', action="store", type=int, default=1,
                          help='return top k most likely classes')

predict_parser.add_argument('--gpu', dest='gpu', action="store", type=str, default='gpu',
                          help='use CUDA/GPU')
args = predict_parser.parse_args()

try:
    probs, classes = icu.predict(args.input[0], args.checkpoint[0], args.top_k, args.gpu)
except AssertionError:
    sys.exit("Found no NVIDIA driver to use CUDA/GPU. Please enter argument '--gpu cpu' !")
except RuntimeError:
    sys.exit("Found no NVIDIA driver to use CUDA/GPU. Please enter argument '--gpu cpu' !")

# get class names
with open(args.category_to_names, 'r') as f:
    cat_to_name = json.load(f)

class_names = []
for c in classes:
    class_names.append(cat_to_name[c])

for i in range(args.top_k):
    print('{:s} - {:.2f}%'.format(class_names[i], probs[i]*100))
