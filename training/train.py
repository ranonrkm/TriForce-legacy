import torch
import torch.nn as nn

from termcolor import colored

def compute_metrics(p): 
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    logits = p.predictions 
    logits = logits[0][:, : -1, :] 
    logits = torch.tensor(logits) 
    logits = logits.view(-1, logits.shape[-1]) 
    
    labels = p.label_ids 
    labels = labels[:, 1:] 
    labels = torch.tensor(labels) 
    labels = labels.view(-1) 
    
    probs = torch.softmax(torch.tensor(logits), dim = -1) 
    loss = nn.CrossEntropyLoss()(torch.tensor(logits), torch.tensor(labels)).item() 
    perplexity = torch.exp(torch.tensor(loss)).item() 

    pred = torch.argmax(probs, dim = -1) 

    output = {
        'accuracy': accuracy_score(labels, pred), 
        'perplexity': perplexity,
    } 
    print(colored(output, "red")) 
    return output