from transform import SelectionSequentialTransform 
from transformers import BertModel, BertConfig,  BertTokenizerFast
from encoder import PolyEncoder 
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
import argparse
import pandas as pd 
import os 
import pickle


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model", default='ckpt/pretrained/bert-small-uncased', type=str)
    parser.add_argument("--text", default='안녕하십니까', type=str)  # 한 줄에 '몇 번 버스를 타시면 됩니다.\n' 이렇게 한문장 저장
    parser.add_argument("--max_response_length", default=128, type=int)
    parser.add_argument("--emb_dir", default='path/to/emb.df', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    
    args = parser.parse_args()
    print(args)

    bert_config = BertConfig.from_json_file(os.path.join(args.bert_model, 'config.json'))

    previous_model_file = os.path.join(args.bert_model, "pytorch_model.bin")
    print('Loading parameters from', previous_model_file)
    model_state_dict = torch.load(previous_model_file, map_location="cpu")
    bert = BertModel.from_pretrained(args.bert_model, state_dict=model_state_dict)

    model = PolyEncoder(bert_config, bert=bert, poly_m=16)

    tokenizer = BertTokenizerFast.from_pretrained(args.bert_model, do_lower_case=True, clean_text=False)
    response_transform = SelectionSequentialTransform(tokenizer=tokenizer, max_len=args.max_response_length)

    dataset = OneSentenceDataset(args.text_path, response_transform, mode='poly')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.resize_token_embeddings(len(tokenizer)) 
    model.to(device)

    embeddings = []
    with torch.no_grad():
        for ids, masks in tqdm(dataloader): 
            ids = ids[:, 0, :].unsqueeze(1)
            masks = masks[:, 0, :].unsqueeze(1)
            batch_size, res_cnt, seq_length = ids.shape
            ids = ids.view(-1, seq_length)
            masks = masks.view(-1, seq_length)
            cand_emb = model.bert(ids, masks)[0][:,0,:] # [bs, dim]
            embeddings.append(cand_emb.to('cpu'))

    emb_df = pd.DataFrame({'text':dataset.data_source, 'embedding' : embeddings})
    
    with open(os.path.join(args.output_dir, 'cand_embs.pickle'), 'wb') as f: 
        pickle.dump(emb_df, f)