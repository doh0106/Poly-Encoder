import torch
from transform import SelectionSequentialTransform, SelectionJoinTransform

class Call_Center(): 
    def __init__(self, model, tokenizer, cand_embs_df, device): 
        self.context_transform = SelectionJoinTransform(tokenizer=tokenizer, max_len=128)
        self.model = model
        self.cand_embs_df = cand_embs_df 
        self.device = device
    def inference(self, query): 
        def context_input(context):
            context_input_ids, context_input_masks = self.context_transform(context)
            contexts_token_ids_list_batch, contexts_input_masks_list_batch = [context_input_ids], [context_input_masks]
            long_tensors = [contexts_token_ids_list_batch, contexts_input_masks_list_batch]
            contexts_token_ids_list_batch, contexts_input_masks_list_batch = (torch.tensor(t, dtype=torch.long, device=self.device) for t in long_tensors)
            return contexts_token_ids_list_batch, contexts_input_masks_list_batch
        
        def embs_gen(contexts_token_ids_list_batch, contexts_input_masks_list_batch):
            ctx_out = self.model.bert(contexts_token_ids_list_batch, contexts_input_masks_list_batch)[0]  # [bs, length, dim]
            poly_code_ids = torch.arange(self.model.poly_m, dtype=torch.long).to(self.device)
            poly_code_ids = poly_code_ids.unsqueeze(0).expand(1, self.model.poly_m)
            poly_codes = self.model.poly_code_embeddings(poly_code_ids) # [bs, poly_m, dim]
            embs = self.model.dot_attention(poly_codes, ctx_out, ctx_out) # [bs, poly_m, dim]
            return embs
        
        def score(embs, cand_emb):
            ctx_emb = self.model.dot_attention(cand_emb, embs, embs) # [bs, res_cnt, dim]
            dot_product = (ctx_emb*cand_emb).sum(-1)
            return dot_product

        with torch.no_grad(): 
            answers = self.cand_embs_df
            cand_embs = torch.stack(answers['embedding'].tolist(), dim=1).to(self.device)
            embs = embs_gen(*context_input([query]))
            embs = embs.to(self.device)
            s = score(embs, cand_embs)
            idx = int(s.argmax(1)[0])
            return answers['text'][idx]
