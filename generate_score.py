from dataset import *
from model import *
from torch.utils.data import DataLoader
import torch
import os
from tqdm import tqdm
import argparse
torch.multiprocessing.set_start_method('spawn', force=True)

def init():
    parser = argparse.ArgumentParser("load model scores")
    parser.add_argument('--model_folder', type=str, help="directory for pretrained model",
                        default='./models/try/')
    parser.add_argument('-n', '--model_name', type=str, help="the name of the model",
                        required=False, default='w2v2_LCNN')
    parser.add_argument('-s', '--score_dir', type=str, help="folder path for writing score",
                        default='./scores')
    parser.add_argument("-t", "--task", type=str, help="which dataset you would liek to score on",
                        required=False, default='19eval')
    parser.add_argument("--gpu", type=str, help="GPU index", default="2")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    if '19' in args.task:
        args.out_score_dir = "./scores"
    else:
        args.out_score_dir = args.score_dir

    return args


def test_on_ASVspoof2019(task, feat_model_path, output_score_path, model_name):
    dirname = os.path.dirname
    basename = os.path.splitext(os.path.basename(feat_model_path))[0]
    if "checkpoint" in dirname(feat_model_path):
        dir_path = dirname(dirname(feat_model_path))
    else:
        dir_path = dirname(feat_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(feat_model_path)
    # model = torch.nn.DataParallel(model, list(range(torch.cuda.device_count())))  # for multiple GPUs
    #loss_model = torch.load(loss_model_path)
    test_set = ASVspoof2019("LA", '/home/chenghaonan/xieyuankun/data/asv2019/preprocess_xls-r', 'eval', "xls-r", pad_chop=False)
    testDataLoader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8)
    model.eval()

    if '19' in task:
        txt_file_name = os.path.join(output_score_path, model_name + '_' + task + '_score.txt')
    else:
        txt_dir = os.path.join(output_score_path, model_name + '_' + task)
        if not os.path.exists(txt_dir):
            os.makedirs(txt_dir)
        txt_file_name = os.path.join(txt_dir, 'score.txt')

    with open(txt_file_name, 'w') as cm_score_file:
        for i, data_slice in enumerate(tqdm(testDataLoader)):
            w2v2, audio_fn, labels = data_slice[0], data_slice[1], data_slice[3]
            w2v2 = w2v2.transpose(2, 3).to(device)
            labels = labels.to(device)
            feats, w2v2_outputs = model(w2v2)
            score = -F.softmax(w2v2_outputs)[:, 0]
            if '19' in task:
                for j in range(labels.size(0)):
                    print(score[j].item())
                    cm_score_file.write('%s %s %s\n' % (
                    audio_fn[j], -score[j].item(), "spoof" if labels.data.cpu() == torch.tensor([1]) else "bonafide"))
            else:
                for j in range(labels.size(0)):
                    cm_score_file.write('%s %s\n' % (audio_fn[j], -score[j].item()))


if __name__ == "__main__":
    args = init()
    model_dir = os.path.join(args.model_folder)
    model_path = os.path.join(model_dir, "anti-spoofing_feat_model.pt")
    test_on_ASVspoof2019(args.task, model_path, args.score_dir, args.model_name)