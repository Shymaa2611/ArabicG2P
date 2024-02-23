from Data.dataset import prepare_dataset
from models.transformers import Seq2SeqTransformer
from macros import *
from trainer import train

if __name__=="__main__":
   data_dir="Data//data.csv"
   train_data,test_data,all_data,source_vocab_size,target_vocab_size=prepare_dataset(data_dir)
   model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,NHEAD, source_vocab_size, target_vocab_size, FFN_HID_DIM)   
   model = model.to(DEVICE)
   optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
   train(model,optimizer,train_data,test_data)

