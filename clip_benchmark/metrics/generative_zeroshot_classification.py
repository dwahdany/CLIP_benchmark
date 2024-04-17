import torch
import torch.nn.functional as F

import logging
from contextlib import suppress

import torch
import torch.nn.functional as F
from tqdm import tqdm

from sklearn.metrics import classification_report, balanced_accuracy_score

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

def build_prompts(model, classnames, templates, device, amp=True):
    autocast = torch.cuda.amp.autocast if amp else suppress
    prompts = []
    for classname in classnames:
        if type(templates) == dict:
            # class-specific prompts (e.g., CuPL https://arxiv.org/abs/2209.03320)
            texts = templates[classname]
        elif type(templates) == list:
            # generic prompts tht are specialized for each class by replacing {c} with the class name
            texts = [template.format(c=classname) for template in templates]
        else:
            raise ValueError("templates must be a list or a dict")
        prompts.append(texts)
    return prompts

def get_any(dic, keys, default=None):
    for k in keys:
        if k in dic:
            return dic[k]
    return default

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """
    Compute top-k accuracy

    output: torch.Tensor
        shape (N, C) where N is the number of examples, C the number of classes.
        these are the logits.
    
    target: torch.Tensor
        shape (N,) where N is the number of examples. Groundtruth class id of each example.
    
    topk: tuple
        which topk to compute, e.g., topk=(1,5) will compute top-1 and top-5 accuracies
    
    Returns
    -------
    
    list of top-k accuracies in the same order as `topk`
    """
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    n = len(target)
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) / n for k in topk]


def run_classification(model, prompts, dataloader, device, tokenizer=None, amp=True, normalize=False, normalizer=None, distributed=False, normalize_coef=1, prompt_batch_size=64, pad_id=0, verbose=False):
    """
    Run zero-shot classifcation

    model: torch.nn.Module
        CLIP-like model with `encode_image` and `encode_text`
    
    classifier: torch.Tensor
        obtained from the function `zero_shot_classifier`
    
    dataloader: torch.utils.data.Dataloader 
    
    Returns
    -------
    (pred, true)  where
        - pred (N, C) are the logits
        - true (N,) are the actual classes
    """
    autocast = torch.cuda.amp.autocast if amp else suppress

    nb_classes = len(prompts)
    nb_templates = len(prompts[0])
    tokenized_prompts = tokenizer([ct for cl in prompts for ct in cl])
    _, context_length = tokenized_prompts.shape
    tokenized_prompts = tokenized_prompts.view(nb_classes, nb_templates, context_length)

    if normalize and normalizer:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_id = normalizer
        lm_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map=device, trust_remote_code=True)
        lm_tokenizer =  AutoTokenizer.from_pretrained(model_id)
        lm_tokenizer.pad_token = lm_tokenizer.eos_token
        if verbose:
            print(f"Using {model_id} to normalize")
    pred = []
    true = []
    nb = 0
    nb_classes, nb_templates, context_length = tokenized_prompts.shape
    max_text_len = (tokenized_prompts==pad_id).float().argmax(dim=2).max()
    tokenized_prompts = tokenized_prompts[:, :, 0:max_text_len]
    tokenized_prompts = tokenized_prompts.to(device)
    tokenized_prompts_ = tokenized_prompts.view(nb_classes * nb_templates, -1)
    if normalize:
        if verbose:
            print("Calculating priors for normalization...")
        if normalizer:
            prompts_ = [p for ps in prompts for p in ps]
            prompts_ = lm_tokenizer.batch_encode_plus(
                prompts_, padding=True, return_tensors="pt", max_length=77, pad_to_max_length=True).input_ids
            prompts_ = prompts_.to(device)
            priors_all = []
            for i in tqdm(range(0, prompts_.shape[0], prompt_batch_size)):
                output = lm_model(prompts_[i:i+prompt_batch_size, 0:-1])
                target = prompts_[i:i+prompt_batch_size, 1:]
                priors = -F.cross_entropy(output.logits.transpose(1, 2), target, reduction="none", ignore_index=lm_tokenizer.pad_token_id).sum(dim=1).data.cpu()
                priors_all.append(priors)
            priors = torch.cat(priors_all, 0)
            priors = priors.view(1, nb_classes, nb_templates).to(device)
        else:
            all_scores = []
            for i in tqdm(range(0, tokenized_prompts_.shape[0], prompt_batch_size)):
                input_text = tokenized_prompts_[i:i+bs, 0:-1]
                nt = len(input_text)
                # out_text = tokenized_prompts_[i:i+bs, 1:]
                # logits, _ = model._encode_text(input_text, image_embs=None)
                out = model.forward(
                    image=None,
                    image_embs=None,
                    text=tokenized_prompts_[i:i+prompt_batch_size],
                )
                logits = get_any(out, ["logits_text", "logits"])
                labels = get_any(out, ["labels_text", "labels"])
                scores = score_aligned(logits, labels).cpu()
                all_scores.append(scores)
            priors = torch.cat(all_scores)
            priors = priors.view(1, nb_classes, nb_templates).to(device)
        if verbose:
            print(f"Priors shape: {priors.shape}")
    else:
        priors = None
    
    with torch.no_grad(), autocast():
        for images, target in tqdm(dataloader):
            images = images.to(device)
            target = target.to(device)
            scores_batch = []
            _, image_embs = model._encode_image(images)
            nim, lim, dim = image_embs.shape
            for i in range(0, nb_classes*nb_templates, prompt_batch_size):
                texts = tokenized_prompts_[i:i+prompt_batch_size, :]
                ntext, ltext = texts.shape
                image_embs_p = image_embs.view(nim, 1, lim, dim).repeat(1, ntext, 1, 1).view(nim*ntext, lim, dim)
                texts_p = texts.view(1, ntext, ltext).repeat(nim, 1, 1).view(nim*ntext, ltext)
                out = model.forward(
                    text=texts_p,
                    image_embs=image_embs_p,
                )
                logits = get_any(out, ["logits_text", "logits"])
                labels = get_any(out, ["labels_text", "labels"])
                scores = score_aligned(logits, labels)
                scores = scores.view(nim, ntext)
                scores_batch.append(scores.float())
            scores = torch.cat(scores_batch, dim=1)
            scores = scores.view(nim, nb_classes, nb_templates)
            if normalize:
                scores = scores - normalize_coef * priors
            # score(class i) = sum score prompts for class i
            scores = scores.sum(dim=-1)
            true.append(target.cpu())
            pred.append(scores.cpu())
    pred = torch.cat(pred)
    true = torch.cat(true)
    if distributed:
        # barrier
        torch.distributed.barrier()
        total = torch.tensor([len(pred)]).to(device)
        torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)
        total = total.item()
        pred = all_gather_nd(pred.to(device)).cpu()
        true = all_gather_nd(true.to(device)).cpu()
    return pred, true


def all_gather_nd(tensor):
    """
    Gathers tensor arrays of different lengths in a list.
    The length dimension is 0. This supports any number of extra dimensions in the tensors.
    All the other dimensions should be equal between the tensors.

    Args:
        tensor (Tensor): Tensor to be broadcast from current process.

    Returns:
        (Tensor): output list of tensors that can be of different sizes
    """
    world_size = torch.distributed.get_world_size()
    local_size = torch.tensor(tensor.size(), device=tensor.device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    torch.distributed.all_gather(all_sizes, local_size)

    max_length = max(size[0] for size in all_sizes)

    length_diff = max_length.item() - local_size[0].item()
    if length_diff:
        pad_size = (length_diff, *tensor.size()[1:])
        padding = torch.zeros(pad_size, device=tensor.device, dtype=tensor.dtype)
        tensor = torch.cat((tensor, padding))

    all_tensors_padded = [torch.zeros_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(all_tensors_padded, tensor)
    all_tensors = []
    for tensor_, size in zip(all_tensors_padded, all_sizes):
        all_tensors.append(tensor_[:size[0]])
    return torch.cat(all_tensors)

def average_precision_per_class(scores, targets):
    """
    Compute average precision  for each class
    this metric is used for multi-label classification
    see explanations here https://fangdahan.medium.com/calculate-mean-average-precision-map-for-multi-label-classification-b082679d31be
    Code is adapted from https://github.com/pytorch/tnt/blob/master/torchnet/meter/meter.py, thanks to the authors of `tnt`.

    Parameters
    ----------

    scores: torch.Tensor
        logits, of shape (N,C) where N is the number of examples, C the number of classes
    
    targets: torch.Tensor
        one-hot vectors of groundtruth targets (N, C), where N is the number of examples, C is the
        number of classes
    
    Returns
    -------

    torch.Tensor of shape (C,) of avereage precision for each class, where C is     
    the number of classes.
    
    """
    ap = torch.zeros(scores.size(1))
    rg = torch.arange(1, scores.size(0) + 1).float()
    # compute average precision for each class
    for k in range(scores.size(1)):
        # sort scores
        scores_k = scores[:, k]
        targets_k = targets[:, k]
        _, sortind = torch.sort(scores_k, 0, True)
        truth = targets_k[sortind]
        tp = truth.float().cumsum(0)
        # compute precision curve
        precision = tp.div(rg)
        # compute average precision
        ap[k] = precision[truth.bool()].sum() / max(float(truth.sum()), 1)
    return ap


def evaluate(model, dataloader, tokenizer, classnames, templates, device, amp=True, verbose=False, normalize=False, normalizer=None, normalize_coef=1, distributed=False, prompt_batch_size=64):
    """
    Run zero-shot classification and evaluate the metrics

    Parameters
    ----------

    model: torch.nn.Module
        CLIP-like model with `encode_image` and `encode_text`
    
    dataloader: torch.utils.data.Dataloader

    tokenizer: text tokenizer

    classnames: list of str
        class names
    
    templates: list of str
        templates to use for zero-shot classification
    
    device: cpu/cuda

    amp: whether to use automatic mixed precision

    verbose: whether to use verbose model

    Returns
    -------

    dict of classification metrics
    """
   
    prompts = build_prompts(model, classnames, templates, device, amp=amp)
    logits, target = run_classification(
        model, prompts, dataloader, device, 
        tokenizer=tokenizer,
         amp=amp, 
        normalize=normalize, 
        normalizer=normalizer,
        normalize_coef=normalize_coef,
        distributed=distributed,
        prompt_batch_size=prompt_batch_size,
        verbose=verbose,
    )
    is_multilabel = (len(target.shape) == 2)

    if is_multilabel:
        if verbose:
            print("Detected a multi-label classification dataset")
        # Multiple labels per image, multiple classes on the dataset
        ap_per_class = average_precision_per_class(logits, target)
        if verbose:
            for class_name, ap in zip(dataloader.dataset.classes, ap_per_class.tolist()):
                print(f"Class: {class_name}, AveragePrecision: {ap}")
        return {"mean_average_precision": ap_per_class.mean().item()}
    else:
        # Single label per image, multiple classes on the dataset
        # just compute accuracy and mean_per_class_recall

        pred = logits.argmax(axis=1)
        # measure accuracy
        if len(dataloader.dataset.classes) >= 5:
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        else:
            acc1, = accuracy(logits, target, topk=(1,))
            acc5 = float("nan") 
        mean_per_class_recall = balanced_accuracy_score(target, pred)
        if verbose:
            print(classification_report(target, pred, digits=3))
        return {"acc1": acc1, "acc5": acc5, "mean_per_class_recall": mean_per_class_recall}
