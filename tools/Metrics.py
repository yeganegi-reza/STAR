import torch

def get_recall(indices, targets): 
    """
    Calculates the recall score for the given predictions and targets
    """
    targets = targets.view(-1, 1).expand_as(indices)
    hits = (targets == indices)
    hits = torch.sum(hits, 1)
    n_hits = torch.sum(hits).cpu().item()
    if n_hits == 0:
        return (hits > 0), 0
    recall = n_hits 
    return (hits > 0), recall


def get_mrr(indices, targets):
    """
    Calculates the MRR score for the given predictions and targets
    """
    tmp = targets.view(-1, 1)
    targets = tmp.expand_as(indices)
    hits = (targets == indices).nonzero()
    ranks = hits[:, -1] + 1
    ranks = ranks.float()
    rranks = torch.reciprocal(ranks)
    mrr = torch.sum(rranks).data
    return mrr.cpu().numpy()


def calc(indices, targets, k=20):
    """
    compute Recall@K, MRR@K scores.
    """
    _ , indices = torch.topk(indices, k, -1)
    _, recall = get_recall(indices, targets)
    mrr = get_mrr(indices, targets)
    return recall, mrr
