import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
text_transform={}
token_transform = {}
vocab_transform = {}
input  ='grapheme'
output ='phoneme'
NUM_EPOCHS = 5
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 5
NUM_DECODER_LAYERS = 5
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)



