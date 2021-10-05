import torch
import os
from options import TestOptions
from dataset import Dataset
from model import TwinsNet
from saver import save_imgs

def main():
    parser = TestOptions()
    args = parser.parse()

    # data loader
    print("\n----------load dataset------------")
    dataset = Dataset(args)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=args.n_threads, shuffle=True)

    # model
    print("\n----------load model---------------")
    model = TwinsNet(args)
    model.set_gpu(args.gpu)
    model.resume(args.resume, train=False)
    model.eval()

    # dir
    result_dir = os.path.join(args.result_dir, args.name)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    # test
    print("\n---------test-----------")
    for index1, (input_a, _) in enumerate(test_loader):
        print("%d/%d" % (index1, len(test_loader)))
        input_a = input_a.cuda()
        imgs = [input_a]
        names = ["input"]
        for index2, (_, input_b) in enumerate(test_loader):
            if index2 == args.num:
                break
            input_b = input_b.cuda()
            with torch.no_grad():
                output = model.test_transfer(input_a, input_b)
            imgs.append(output)
            names.append("output_%d" % index2)
        save_imgs(imgs, names, os.path.join(result_dir, "{}".format(index1)))

if __name__ == "__main__":
    main()