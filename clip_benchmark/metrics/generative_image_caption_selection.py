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
    Evaluate the model on the given dataset.
    The task has N instances, each instance has I images and C captions.
    For each instance, the goal is to find the correct image for each caption and the correct caption for each image.
    This is done by computing the similarities between each image and each caption.
    This procedure is used to evaluate the models on Winoground and SugarCrepe.

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
    
    dict of accuracy metrics
    """
    autocast = torch.cuda.amp.autocast if amp else suppress
    image_score = []
    text_score = []
    score = []
    for batch_images, batch_texts in tqdm(dataloader):
        if len(batch_images.shape) == 4:
            B, C, H, W = batch_images.shape
            batch_images = batch_images.view(B, 1, C, H, W)
        # batch_images: B, nb_images_per_instance, C, H, W
        # batch_texts: B, nb_captions_per_instance
        
        B, nim, C, H, W = batch_images.shape
        nt = len(batch_texts[0])
        batch_images = batch_images.to(device)
        batch_images_ = batch_images.view(B*nim, C, H, W) # B*nim, C, H, W
        # tokenize all texts in the batch
        batch_texts_tok_ = tokenizer([text for i, texts in enumerate(batch_texts) for text in texts]).to(device)
        batch_texts_tok_ = batch_texts_tok_.view(B, nt, -1)
        # compute the embedding of images and texts
        with torch.no_grad(), autocast():
            _, batch_images_emb = model._encode_image(batch_images_)
            bs, seqlen, hidden = batch_images_emb.shape
            batch_images_emb = batch_images_emb.view(B, nim, seqlen, hidden)
            gt = torch.arange(min(nim, nt)).to(device)
        for i in range(B):
            # iteratve over instances
            image_emb = batch_images_emb[i]
            texts = batch_texts_tok_[i]
            max_text_len = (texts==0).float().argmax(dim=-1).max().item()
            texts = texts[:, :max_text_len]
            if torch.any(torch.isnan(image_emb)):
                print("Detected nans in image embs..")
                return {"acc": 0.0}
            nim, lim, dim = image_emb.shape
            ntext, ltext = texts.shape
            with torch.no_grad(), autocast():
                image_embs_p = image_emb.view(nim, 1, lim, dim).repeat(1, ntext, 1, 1).view(nim*ntext, lim, dim)
                texts_p = texts.view(1, ntext, ltext).repeat(nim, 1, 1).view(nim*ntext, ltext)
                
                input_text = texts_p[:, 0:-1]
                out_text = texts_p[:, 1:]
                _, input_text_embs_p = model._encode_text(input_text)
                logits, _ = model._encode_text(input_text, image_embs_p)

                scores = score_aligned(logits, out_text)
                scores = scores.view(nim, ntext)
                if torch.any(torch.isnan(scores)):
                    pred_image_is_correct = False
                    pred_text_is_correct = False
                else:
                    # i-th image should be matched to the i-th text
                    image_closest_text = scores.argmax(dim=1)
                    # -- deal with ties
                    if ntext > 1:
                        val = scores[:, 0:1]
                        all_equal = ((scores==val).sum(dim=1) == ntext)
                        image_closest_text[all_equal] = -1

                    # i-th text should be matched to the i-th image
                    text_closest_image = scores.argmax(dim=0)
                    # -- deal with ties
                    val = scores[0:1, :]
                    if nim > 1:
                        all_equal = ((scores==val).sum(dim=0) == nim)
                        text_closest_image[all_equal] = -1

                    image_closest_text = image_closest_text[:len(gt)]
                    text_closest_image = text_closest_image[:len(gt)]

                    pred_text_is_correct = (image_closest_text==gt).all().item()
                    pred_image_is_correct = (text_closest_image==gt).all().item()
            all_correct = pred_text_is_correct and pred_image_is_correct
            image_score.append(pred_image_is_correct)
            text_score.append(pred_text_is_correct)
            score.append(all_correct)
    metrics = {}
    metrics["image_acc"] = torch.Tensor(image_score).float().mean().item()
    metrics["text_acc"] = torch.Tensor(text_score).float().mean().item()
    metrics["acc"] = torch.Tensor(score).float().mean().item()
    return metrics
