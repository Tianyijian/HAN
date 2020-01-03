import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from model import HierarchialAttentionNetwork
from utils import *
from datasets import HANDataset
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.tensorboard import SummaryWriter

# Data parameters
data_folder = '/users5/yjtian/tyj/demo/HAN/data'
word2vec_file = os.path.join(data_folder, 'word2vec_model')  # path to pre-trained word2vec embeddings
with open(os.path.join(data_folder, 'word_map.json'), 'r') as j:
    word_map = json.load(j)

# Model parameters
n_classes = len(label_map)
word_rnn_size = 50  # word RNN size
sentence_rnn_size = 50  # character RNN size
word_rnn_layers = 1  # number of layers in character RNN
sentence_rnn_layers = 1  # number of layers in word RNN
word_att_size = 100  # size of the word-level attention layer (also the size of the word context vector)
sentence_att_size = 100  # size of the sentence-level attention layer (also the size of the sentence context vector)
dropout = 0.3  # dropout
fine_tune_word_embeddings = True  # fine-tune word embeddings?

# Training parameters
start_epoch = 0  # start at this epoch
batch_size = 512  # batch size
lr = 1e-3  # learning rate
momentum = 0.9  # momentum
workers = 4  # number of workers for loading data in the DataLoader
epochs = 8  # number of epochs to run
grad_clip = None  # clip gradients at this value
print_freq = 100  # print training or validation status every __ batches
checkpoint = None  # path to model checkpoint, None if none

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
writer = SummaryWriter(log_dir="logs")
global_step = 0


def main():
    """
    Training and validation.
    """
    global checkpoint, start_epoch, word_map

    iter = 5
    res = {"best_eval_acc": [], "best_eval_f1": [], "best_eval_step": []}
    for i in range(1, 1 + iter):
        print("=" * 10 + "ROUND " + str(i) + "=" * 10)

        # DataLoaders
        train_loader = torch.utils.data.DataLoader(HANDataset(data_folder, 'train'), batch_size=batch_size,
                                                   shuffle=True, num_workers=workers, pin_memory=True)
        # Load test data
        test_loader = torch.utils.data.DataLoader(HANDataset(data_folder, 'test'), batch_size=batch_size, shuffle=False,
                                                  num_workers=workers, pin_memory=True)


        # Initialize model or load checkpoint
        if checkpoint is not None:
            checkpoint = torch.load(checkpoint)
            model = checkpoint['model']
            optimizer = checkpoint['optimizer']
            word_map = checkpoint['word_map']
            start_epoch = checkpoint['epoch'] + 1
            print(
                '\nLoaded checkpoint from epoch %d.\n' % (start_epoch - 1))
        else:
            # embeddings, emb_size = load_word2vec_embeddings(word2vec_file, word_map)  # load pre-trained word2vec embeddings

            # embeddings, emb_size = load_glove_w2v(word_map)  # load pre-trained word2vec embeddings

            emb_size = 200
            # embeddings = torch.FloatTensor(len(word_map), emb_size)
            # init_embedding(embeddings)

            model = HierarchialAttentionNetwork(n_classes=n_classes,
                                                vocab_size=len(word_map),
                                                emb_size=emb_size,
                                                word_rnn_size=word_rnn_size,
                                                sentence_rnn_size=sentence_rnn_size,
                                                word_rnn_layers=word_rnn_layers,
                                                sentence_rnn_layers=sentence_rnn_layers,
                                                word_att_size=word_att_size,
                                                sentence_att_size=sentence_att_size,
                                                dropout=dropout)
            # model.sentence_attention.word_attention.init_embeddings(
            #     embeddings)  # initialize embedding layer with pre-trained embeddings
            model.sentence_attention.word_attention.fine_tune_embeddings(fine_tune_word_embeddings)  # fine-tune
            optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

        # Loss functions
        criterion = nn.CrossEntropyLoss()

        # Move to device
        model = model.to(device)
        criterion = criterion.to(device)

        best_acc = 0.0
        best_f1 = 0.0
        best_step = 0

        # Epochs
        for epoch in range(start_epoch, epochs):
            # One epoch's training
            eval_acc, eval_f1, eval_step = train(train_loader=train_loader,
                                                 test_loader=test_loader,
                                                 model=model,
                                                 criterion=criterion,
                                                 optimizer=optimizer,
                                                 epoch=epoch)

            if eval_acc > best_acc:
                best_acc = eval_acc
                best_f1 = eval_f1
                best_step = eval_step

            # Decay learning rate every epoch
            # adjust_learning_rate(optimizer, 0.1)

            # Save checkpoint
            # save_checkpoint(epoch, model, optimizer, word_map)

        res["best_eval_acc"].append(best_acc)
        res["best_eval_f1"].append(best_f1)
        res["best_eval_step"].append(best_step)

    print("=" * 20 + "TRAINING FINISHED" + "=" * 20)
    print("avg acc: %f" % (float(np.sum(res["best_eval_acc"])) / len(res["best_eval_acc"])))
    print("avg f1: %f" % (float(np.sum(res["best_eval_f1"])) / len(res["best_eval_f1"])))
    print(" ".join(["{}: {}".format(key, str(res[key])) for key in res]))
    writer.close()


def train(train_loader, test_loader, model, criterion, optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: cross entropy loss layer
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    global global_step
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time per batch
    data_time = AverageMeter()  # data loading time per batch
    losses = AverageMeter()  # cross entropy loss
    accs = AverageMeter()  # accuracies

    start = time.time()
    best_acc = 0.0
    best_f1 = 0.0
    best_step = 0
    # Batches
    for i, (documents, sentences_per_document, words_per_sentence, labels) in enumerate(train_loader):

        data_time.update(time.time() - start)

        documents = documents.to(device)  # (batch_size, sentence_limit, word_limit)
        sentences_per_document = sentences_per_document.squeeze(1).to(device)  # (batch_size)
        words_per_sentence = words_per_sentence.to(device)  # (batch_size, sentence_limit)
        labels = labels.squeeze(1).to(device)  # (batch_size)

        optimizer.zero_grad()
        model.train()

        # Forward prop.
        scores, word_alphas, sentence_alphas = model(documents, sentences_per_document,
                                                     words_per_sentence)  # (n_documents, n_classes), (n_documents, max_doc_len_in_batch, max_sent_len_in_batch), (n_documents, max_doc_len_in_batch)

        # Loss
        loss = criterion(scores, labels)  # scalar

        # Back prop.
        # optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update
        optimizer.step()

        # Find accuracy
        _, predictions = scores.max(dim=1)  # (n_documents)
        correct_predictions = torch.eq(predictions, labels).sum().item()
        accuracy = correct_predictions / labels.size(0)

        # Keep track of metrics
        losses.update(loss.item(), labels.size(0))
        batch_time.update(time.time() - start)
        accs.update(accuracy, labels.size(0))

        global_step += 1
        writer.add_scalar("loss", loss.item(), global_step)
        writer.add_scalar("batch_acc", accuracy, global_step)
        start = time.time()

        # Print training status
        if i % print_freq == 0:
            eval_acc, eval_f1 = eval(test_loader, model)
            writer.add_scalar("eval_acc", eval_acc, global_step)
            writer.add_scalar("eval_f1", eval_f1, global_step)

            if eval_acc > best_acc:
                best_acc = eval_acc
                best_f1 = eval_f1
                best_step = i + epoch * len(train_loader)

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t'
                  'Eval Acc {eval_acc:.4f}, f1 {eval_f1:.4f}'.format(epoch, i, len(train_loader),
                                                                     batch_time=batch_time,
                                                                     data_time=data_time, loss=losses,
                                                                     acc=accs, eval_acc=eval_acc, eval_f1=eval_f1))

    return best_acc, best_f1, best_step


def eval(test_loader, model):
    model.eval()

    # Evaluate in batches
    pred = []
    label = []
    for i, (documents, sentences_per_document, words_per_sentence, labels) in enumerate(test_loader):
        documents = documents.to(device)  # (batch_size, sentence_limit, word_limit)
        sentences_per_document = sentences_per_document.squeeze(1).to(device)  # (batch_size)
        words_per_sentence = words_per_sentence.to(device)  # (batch_size, sentence_limit)
        labels = labels.squeeze(1).to(device)  # (batch_size)

        # Forward prop.
        scores, word_alphas, sentence_alphas = model(documents, sentences_per_document,
                                                     words_per_sentence)  # (n_documents, n_classes), (n_documents, max_doc_len_in_batch, max_sent_len_in_batch), (n_documents, max_doc_len_in_batch)

        # Find accuracy
        _, predictions = scores.max(dim=1)  # (n_documents)
        pred.extend(predictions.cpu().data.numpy())
        label.extend(labels.cpu().data.numpy())

    acc = accuracy_score(label, pred)
    f1 = f1_score(label, pred, average="macro")
    return acc, f1


if __name__ == '__main__':
    main()
