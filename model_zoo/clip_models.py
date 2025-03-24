import clip
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from open_clip.transformer import text_global_pool




class CLIPWrapper(torch.nn.Module):
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device
    
    #@torch.no_grad()
    def get_text_embeddings(self, texts, text_batch_size=256, normalize=False):
        num_text = len(texts)
        text_embeds = []
        #tqdm_loader = tqdm(range(0, num_text, text_batch_size))
        #tqdm_loader.set_description("Computing text embeddings")
        for i in range(0, num_text, text_batch_size):  #tqdm_loader:
            text = texts[i: min(num_text, i+text_batch_size)]
            text_input = clip.tokenize(text).to(self.device) 
            text_feats = self.model.encode_text(text_input)
            if normalize:
                text_feats = F.normalize(text_feats,dim=-1)      
            text_embeds.append(text_feats)   
        
        text_embeds = torch.cat(text_embeds, dim=0)
        return text_embeds

    @torch.no_grad()
    def get_image_embeddings(self, image_loader, normalize=False, marginalization=False):
        image_embeds = []
        if marginalization:
            for images in image_loader: # it's only one
                image_feats = self.model.encode_image(images.to(self.device))
                image_embeds.append(image_feats)
            return torch.cat(image_embeds, dim=0)
        tqdm_loader = tqdm(image_loader)
        tqdm_loader.set_description("Computing image embeddings")
        for batch in tqdm_loader:
            images = batch["image"]
            image_feats = self.model.encode_image(images.to(self.device))
            if normalize:
                image_feats = F.normalize(image_feats, dim=-1)
            image_embeds.append(image_feats)

        image_embeds = torch.cat(image_embeds, dim=0)
        return image_embeds
    
    @torch.no_grad()
    def get_retrieval_scores_dataset(self, loader):
        captions = loader.dataset.text
        text_embeds = self.get_text_embeddings(captions, normalize=True)
        image_embeds = self.get_image_embeddings(loader, normalize=True)
        scores = image_embeds @ text_embeds.T
        scores = scores.cpu().numpy()
        return scores
    
    @torch.no_grad()
    def get_retrieval_scores_batched(self, joint_loader):
        """Computes the scores for each image_option / caption_option pair in the joint loader.

        Args:
            joint_loader (DataLoader): batches have "image_options" and "caption_options" fields.
            "image_options" is a list of images, and "caption_options" is a list of captions.

        Returns:
            all_scores: A numpy array containing the scores of the shape NxKxL,
            where N is the number of test cases, K is the number of image options per the test case,
            and L is the number of caption options per the test case.
        """
        scores = []
        tqdm_loader = tqdm(joint_loader)
        tqdm_loader.set_description("Computing retrieval scores")
        for batch in tqdm_loader:
            image_options = []
            for i_option in batch["image_options"]:
                image_embeddings = self.model.encode_image(i_option.to(self.device)).cpu().numpy() # B x D
                image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True) # B x D
                image_options.append(np.expand_dims(image_embeddings, axis=1))
            
            caption_options = []
            for c_option in batch["caption_options"]:
                caption_tokenized = torch.cat([clip.tokenize(c) for c in c_option])
                caption_embeddings = self.model.encode_text(caption_tokenized.to(self.device)).cpu().numpy() # B x D
                #caption_embeddings = caption_embeddings / np.linalg.norm(caption_embeddings, axis=1, keepdims=True) # B x D
                caption_options.append(np.expand_dims(caption_embeddings, axis=1))
                
            image_options = np.concatenate(image_options, axis=1) # B x K x D
            caption_options = np.concatenate(caption_options, axis=1) # B x L x D
            batch_scores = np.einsum("nkd,nld->nkl", image_options, caption_options) # B x K x L
            scores.append(batch_scores)
        
        all_scores = np.concatenate(scores, axis=0) # N x K x L
        return all_scores


#Our modification: Add a spatial embedding layer that can be learned so that attention is increased for these words
class ClipWrapperSpatialEmbeds(CLIPWrapper):
    def __init__(self, model, device):
        super().__init__(model, device)
        self.model=model
        self.spatial_vocab = ['on', 'under', 'front', 'behind', 'left', 'right']#["on", "under", "left", "right"]
        self.spatial_embedding = torch.nn.Embedding(len(self.spatial_vocab), model.text_projection.shape[-1]).to(device)
    
    def encode_text_from_open_clip(self, text, spatial_bias_applied, normalize: bool = False):
        cast_dtype = self.model.transformer.get_cast_dtype()

        x = self.model.token_embedding(text).to(cast_dtype)  
        x = x + self.model.positional_embedding.to(cast_dtype)
        x = x + spatial_bias_applied
        x = self.model.transformer(x, attn_mask=self.model.attn_mask)
        

        x = self.model.ln_final(x)  
        
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.model.text_projection

        return F.normalize(x, dim=-1) if normalize else x
    

    def get_text_embeddings(self, texts, text_batch_size=256, normalize=False):
            num_text = len(texts)
            text_embeds = []
            spatial_tokenized = [clip.tokenize(word).squeeze(0)[1] for word in self.spatial_vocab]

            for i in range(0, num_text, text_batch_size):
                text = texts[i: min(num_text, i+text_batch_size)]
                text_input = clip.tokenize(text).to(self.device)

                spatial_mask=[]
                for batched_sentence in text_input:
                    spatial_mask_sentence=[]
                    for token in batched_sentence:
                        token_level_lists=[]
                        for spatial_word_token in spatial_tokenized:
                            if token.item() == spatial_word_token:
                                token_level_lists.append(True)
                            else:
                                token_level_lists.append(False)
                        spatial_mask_sentence.append(token_level_lists)
                    spatial_mask.append(spatial_mask_sentence)

                spatial_mask=torch.tensor(spatial_mask, dtype=torch.float32)
                
                spatial_bias = self.spatial_embedding(torch.arange(len(self.spatial_vocab)).to(self.device)) 
                expanded_bias = spatial_bias.unsqueeze(0).unsqueeze(1)  
                expanded_bias = expanded_bias.expand(spatial_mask.shape[0], spatial_mask.shape[1], -1, -1)  
                masked_spatial_bias=spatial_mask.unsqueeze(-1)*expanded_bias
                spatial_bias_applied=masked_spatial_bias.mean(dim=2)
                spatial_bias_applied= 100*spatial_bias_applied #currently: factor 1000

                text_feats=self.encode_text_from_open_clip(text_input, spatial_bias_applied)
                
                if normalize:
                    text_feats = F.normalize(text_feats, dim=-1)

                text_embeds.append(text_feats)

            text_embeds = torch.cat(text_embeds, dim=0)
            return text_embeds
    
    def forward(
            self,
            image,
            text,
    ):
        image_features = self.model.encode_image(image, normalize=True) if image is not None else None
        text_features = self.get_text_embeddings(text, normalize=True) if text is not None else None

        if self.model.output_dict:
            out_dict = {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.model.logit_scale.exp()
            }
            if self.model.logit_bias is not None:
                out_dict['logit_bias'] = self.model.logit_bias
            return out_dict

        if self.model.logit_bias is not None:
            return image_features, text_features, self.model.logit_scale.exp(), self.model.logit_bias
        
        return image_features, text_features, self.model.logit_scale.exp()



    @torch.no_grad()
    def get_retrieval_scores_batched(self, joint_loader):
        """Computes the scores for each image_option / caption_option pair in the joint loader.

        Args:
            joint_loader (DataLoader): batches have "image_options" and "caption_options" fields.
            "image_options" is a list of images, and "caption_options" is a list of captions.

        Returns:
            all_scores: A numpy array containing the scores of the shape NxKxL,
            where N is the number of test cases, K is the number of image options per the test case,
            and L is the number of caption options per the test case.
        """
        scores = []
        tqdm_loader = tqdm(joint_loader)
        tqdm_loader.set_description("Computing retrieval scores")
        for batch in tqdm_loader:
            image_options = []
            for i_option in batch["image_options"]:
                image_embeddings = self.model.encode_image(i_option.to(self.device)).cpu().numpy() # B x D
                image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True) # B x D
                image_options.append(np.expand_dims(image_embeddings, axis=1))
            
            caption_options = []
            for c_option in batch["caption_options"]:

                caption_tokenized = torch.cat([clip.tokenize(c) for c in c_option])
                caption_embeddings = self.get_text_embeddings(c_option, normalize=True).cpu().numpy() # B x D            
                caption_options.append(np.expand_dims(caption_embeddings, axis=1))
    
            image_options = np.concatenate(image_options, axis=1) # B x K x D
            caption_options = np.concatenate(caption_options, axis=1) # B x L x D
            batch_scores = np.einsum("nkd,nld->nkl", image_options, caption_options) # B x K x L
            scores.append(batch_scores)

        
        all_scores = np.concatenate(scores, axis=0) # N x K x L
        return all_scores
