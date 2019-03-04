import argparse
import image_classifier_utils as icu
import sys

train_parser = argparse.ArgumentParser(description='Create a new deep learning model by choosing the \
                                                    models structure and hidden layer nodes. \
                                                    Afterwards, train the created model with a training dataset.')

train_parser.add_argument('data_dir', nargs="*", action="store", type=str, default="./flowers/",
                          help='directory of training, test and validation datasets')

train_parser.add_argument('--save_dir', dest='save_dir', action="store", default="checkpoint.pth",
                          help='filepath for saving the trained model')

train_parser.add_argument('--arch', dest='arch', action="store", default="vgg11",
                          help='structure for creating the deep learning model')

train_parser.add_argument('--hl1', dest='hidden_layer1', action="store", type=int, default=512,
                          help='nodes of first hidden layer')

train_parser.add_argument('--hl2', dest='hidden_layer2', action="store", type=int, default=256,
                          help='nodes of second hidden layer')

train_parser.add_argument('--hl3', dest='hidden_layer3', action="store", type=int, default=128,
                          help='nodes of third hidden layer')

train_parser.add_argument('--output', dest='output_classes', action="store", type=int, default=102,
                          help='number of output classes')

train_parser.add_argument('--dropout', dest='dropout', action="store", type=float, default=0.2,
                          help='dropout rate')

train_parser.add_argument('--learning_rate', dest='lr', action="store", type=float, default=0.001,
                          help='learning rate for training the model')

train_parser.add_argument('--gpu', dest="gpu", action="store", default="gpu",
                          help='move file to gpu or cpu')

train_parser.add_argument('--epochs', dest="epochs", action="store", type= int, default=5,
                          help='amount of epochs for training the model')

args = train_parser.parse_args()

print('Loading datasets...')
try:
    train_loader, test_loader, valid_loader, train_dataset = icu.load_datasets(args.data_dir[0])
except FileNotFoundError:
    sys.exit("Can't open path " + args.data_dir[0])

try:
    print('Creating neural network...')
    model, optimizer, criterion = icu.nn_create(args.arch, args.dropout, args.hidden_layer1, args.hidden_layer2, args.hidden_layer3, args.output_classes, args.lr, args.gpu)
    print('Training model...')
    model = icu.train_model(model, optimizer, criterion, args.epochs, train_loader, valid_loader, args.gpu)
except AssertionError:
    sys.exit("Found no NVIDIA driver to use CUDA/GPU. Please enter argument '--gpu cpu' !")
except RuntimeError:
    sys.exit("Found no NVIDIA driver to use CUDA/GPU. Please enter argument '--gpu cpu' !")

print('Saving checkpoint...')
save_dir = "./checkpoints/" + args.save_dir
icu.save_checkpoint(model, args.arch, args.hidden_layer1, args.hidden_layer2, args.hidden_layer3, args.output_classes, args.dropout, args.lr, save_dir, train_dataset)

print('Testing model on test dataset...')
acc = icu.calc_accuracy_test(model, test_loader, criterion)

print("Done!\n The model is saved to " + save_dir + " !")
