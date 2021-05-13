import matplotlib.pyplot as plt
import os
import argparse

def parse_acc(f, args):
    lines = f.readlines()
    source_acc = []
    target_acc = []
    stopcode = "Round %d end" % args.round
    for line in lines:
        if 'server' in line:
            acc = float(line[18: 25])
            target_acc.append(acc)
        if stopcode in line:
            break
    return source_acc, target_acc

def parse_loss(f, args):
    lines = f.readlines()
    source_loss = []
    target_loss = []
    stopcode = "Round %d end" % args.round
    trigger = False
    for line in lines:
        if 'begin' in line and not trigger:
            batch = []
            trigger = True
        if 'classifier_loss' in line and trigger:
            loss = float((line.split(',')[1]).split(':')[1])
            batch.append(loss)
        if 'end' in line and trigger:
            loss_ = sum(batch) / (1. * len(batch))
            target_loss.append(loss_)
            trigger = False
        if stopcode in line:
            break
    return source_loss, target_loss

def parse(fn, args):
    with open(fn, 'r') as f:
        if args.yaxis == 0:
            return parse_acc(f, args)
        else:
           return parse_loss(f, args) 

def parse_and_plot(flist, args):
    data = []
    maxy = 0.
    plt.figure(figsize=(16, 12))
    for f in flist:
        _, tmp = parse(f, args)
        p = 0
        for i, _ in enumerate(tmp):
            if tmp[i] > 0.5:
                p = i
                break
        maxy = max(maxy, max(tmp))
        print(f, max(tmp), p + 1)
        label=''
        #label = input('label for %s: ' % f)
        if label == '': label = f
        data.append([tmp, label])
    for item in sorted(data, key=lambda x: x[1]):
        label = item[1].split('/')[-1]
        plt.plot(item[0], label=label, linewidth = 3)
    font = {'family':'Times New Roman', 'size': 40}
    plt.legend(prop = font)
    plt.xlabel('Communication Rounds', fontdict = font)
    if args.yaxis == 0:
        plt.ylim(0.0, 0.7)
        plt.ylabel('Test Accuracy', fontdict = font)
    else: 
        plt.ylim(0.0, maxy * 1.1)
        plt.ylabel('Train Loss', fontdict = font)
    plt.xticks(fontproperties = font['family'], size = font['size'])
    plt.yticks(fontproperties = font['family'], size = font['size'])
    title = '%s %s Similarity %d%%' % (args.data, args.scale, int(100. * args.sim))
    plt.title(title, fontdict = font)
    savename = title + 'acc' if args.yaxis == 0 else title + 'loss'
    plt.savefig(savename + '.pdf')

def plot_folder(args):
    rootdir = args.dir
    list = os.listdir(rootdir)
    flist = []
    for i in range(0,len(list)):
        if list[i][-3:] != 'txt': continue
        path = os.path.join(rootdir,list[i])
        if os.path.isfile(path):
            flist.append(path)
    parse_and_plot(sorted(flist), args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dir', 
        type=str, 
        help='data directory',
    )
    parser.add_argument(
        '--round', 
        type=int, 
        help='maximal round',
    )
    parser.add_argument(
        '--data', 
        type=str, 
        help='dataset',
    )
    parser.add_argument(
        '--scale', 
        type=str, 
        help='Cross-Silo or Cross-Device',
    )
    parser.add_argument(
        '--sim', 
        type=float, 
        help='similarity, 0.0~1.0',
    )
    parser.add_argument(
        '--yaxis', 
        type=int, 
        help='0:accuracy, 1:loss',
    )
    args = parser.parse_args()
    plot_folder(args)