import logging
from contextlib import suppress

import torch
import torch.nn.functional as F
from tqdm import tqdm

def score_aligned(logits, texts):
    # logits: (I, L, V)
    # texts: (I, L)
    I, L, V = logits.shape
    I2, L = texts.shape
    assert I == I2
    lp = logits.log_softmax(dim=-1)
    texts = texts.view(I, L, 1)
    lp = torch.gather(lp, 2, texts)
    lp[texts == 0] = 0
    ce = lp.sum(dim=(1,2))
    return ce

def evaluate(model, dataloader, tokenizer,  device, amp=True):
    """
    Evaluate the model on the given dataset

    Parameters
    ----------
    
    model: torch.nn,Module
        CLIP-like model with `encode_image` and `encode_text`
    
    dataloader: torch.utils.data.Dataloader
        dataloader to use for evaluation

    tokenizer:
        text tokenizer, i.e. convert list of strings to torch.Tensor of integers
    
    device: cpu/cuda

    amp: whether to use automatic mixed precision
    
    Returns
    -------
    
    dict of accuracy metric
    """
    amp = False
    autocast = torch.cuda.amp.autocast if amp else suppress
    preds = []
    for batch_images, batch_texts in tqdm(dataloader):
        batch_images = batch_images.to(device)
        # tokenize all texts in the batch
        batch_texts_tok = tokenizer([text for i, texts in enumerate(batch_texts) for text in texts]).to(device)
        batch_texts_raw = ([text for i, texts in enumerate(batch_texts) for text in texts])#.to(device)

        nb_texts_for_each_image = [len(texts) for texts in batch_texts]

        # compute the embedding of images and texts
        with torch.no_grad(), autocast():
            C, H, W = batch_images.shape[1:]# TODO - make this a parameter

            _, batch_images_emb = model._encode_image(batch_images)
            start = 0
            for i, nb in enumerate(nb_texts_for_each_image):
                end = start + nb
                image_emb = batch_images_emb[i:i+1]
                texts = batch_texts_tok[start:end]
                max_text_len = (texts==0).float().argmax(dim=-1).max().item()
                lens = (texts==0).float().argmax(dim=-1)
                texts = texts[:, :max_text_len]
                if torch.any(torch.isnan(image_emb)):
                    print("Detected nans in image embs..")
                    return {"acc": 0.0}
                nim, lim, dim = image_emb.shape
                ntext, ltext = texts.shape
                image_embs_p = image_emb.view(nim, 1, lim, dim).repeat(1, ntext, 1, 1).view(nim*ntext, lim, dim)
                texts_p = texts.view(1, ntext, ltext).repeat(nim, 1, 1).view(nim*ntext, ltext)
                
                input_text = texts_p[:, 0:-1]
                out_text = texts_p[:, 1:]
                _, input_text_embs_p = model._encode_text(input_text)
                logits = model.text_decoder(image_embs_p, input_text_embs_p)
                scores = score_aligned(logits, out_text)
                scores = scores.view(nim, ntext)
                scores = scores[0]
                if torch.any(torch.isnan(scores)):
                    print("Detected nans..")
                    return {"acc": 0.0}
                pred = scores.argmax().item()
                value = scores[0].item()
                all_equal = torch.all(scores==value).item()
                if all_equal:
                    # tie
                    pred = -1
                start = end 
                preds.append(pred)
    pred = torch.Tensor(preds).long()
    acc = (pred==0).float().mean().item() 
    metrics = {"acc": acc}
    return metrics
