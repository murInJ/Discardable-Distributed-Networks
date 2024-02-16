import torch


def InfoMin_loss(mu, log_var):
    shape = mu.shape
    if len(shape) == 2:
        return torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

    elif len(shape) == 1:
        return -0.5 * torch.mean(1 + log_var - mu ** 2 - log_var.exp())


def InfoMax_loss(x1, x2):
    x1 = x1 / (torch.norm(x1, p=2, dim=1, keepdim=True) + 1e-10)
    x2 = x2 / (torch.norm(x2, p=2, dim=1, keepdim=True) + 1e-10)
    bs = x1.size(0)
    s = torch.matmul(x1, x2.permute(1, 0))
    mask_joint = torch.eye(bs).cuda()
    mask_marginal = 1 - mask_joint

    Ej = (s * mask_joint).mean()
    Em = torch.exp(s * mask_marginal).mean()

    infomax_loss = - (Ej - torch.log(Em))
    return infomax_loss
