# -*- coding: utf-8 -*-
# @Time    : 2023/07/14
# @Author  : Siyang Li
# @File    : loss.py
import numpy as np
import torch as tr
import torch
import torch.nn as nn
import math
import sklearn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Optional, List, Dict, Tuple, Callable, Sequence


def Entropy(input_):
    epsilon = 1e-5
    entropy = -input_ * tr.log(input_ + epsilon)
    entropy = tr.sum(entropy, dim=1)
    return entropy


class CELabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CELabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)

        # 加入mixup之后，原始标签已经是one hot的形式，这里不需要再变换
        # targets = tr.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss


class CELabelSmooth_raw(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CELabelSmooth_raw, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = tr.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss


class ConsistencyLoss(nn.Module):
    """
    Label consistency loss.
    """

    def __init__(self, num_select=2):
        super(ConsistencyLoss, self).__init__()
        self.num_select = num_select

    def forward(self, prob):
        dl = 0.
        count = 0
        for i in range(prob.shape[1] - 1):
            for j in range(i + 1, prob.shape[1]):
                dl += self.jensen_shanon(prob[:, i, :], prob[:, j, :], dim=1)
                count += 1
        return dl / count

    @staticmethod
    def jensen_shanon(pred1, pred2, dim):
        """
        Jensen-Shannon Divergence.
        """
        m = (tr.softmax(pred1, dim=dim) + tr.softmax(pred2, dim=dim)) / 2
        pred1 = F.log_softmax(pred1, dim=dim)
        pred2 = F.log_softmax(pred2, dim=dim)
        return (F.kl_div(pred1, m.detach(), reduction='batchmean') + F.kl_div(pred2, m.detach(),
                                                                              reduction='batchmean')) / 2


# =============================================================DAN Function=============================================
class MultipleKernelMaximumMeanDiscrepancy(nn.Module):
    r"""
    Args:
        kernels (tuple(tr.nn.Module)): kernel functions.
        linear (bool): whether use the linear version of DAN. Default: False

    Inputs:
        - z_s (tensor): activations from the source domain, :math:`z^s`
        - z_t (tensor): activations from the target domain, :math:`z^t`
    """

    def __init__(self, kernels: Sequence[nn.Module], linear: Optional[bool] = False):
        super(MultipleKernelMaximumMeanDiscrepancy, self).__init__()
        self.kernels = kernels
        self.index_matrix = None
        self.linear = linear

    def forward(self, z_s: tr.Tensor, z_t: tr.Tensor) -> tr.Tensor:
        features = tr.cat([z_s, z_t], dim=0)
        batch_size = int(z_s.size(0))
        self.index_matrix = _update_index_matrix(batch_size, self.index_matrix, self.linear).to(z_s.device)

        kernel_matrix = sum([kernel(features) for kernel in self.kernels])  # Add up the matrix of each kernel
        # Add 2 / (n-1) to make up for the value on the diagonal
        # to ensure loss is positive in the non-linear version
        loss = (kernel_matrix * self.index_matrix).sum() + 2. / float(batch_size - 1)

        return loss


def _update_index_matrix(batch_size: int, index_matrix: Optional[tr.Tensor] = None,
                         linear: Optional[bool] = True) -> tr.Tensor:
    r"""
    Update the `index_matrix` which convert `kernel_matrix` to loss.
    If `index_matrix` is a tensor with shape (2 x batch_size, 2 x batch_size), then return `index_matrix`.
    Else return a new tensor with shape (2 x batch_size, 2 x batch_size).
    """
    if index_matrix is None or index_matrix.size(0) != batch_size * 2:
        index_matrix = tr.zeros(2 * batch_size, 2 * batch_size)
        if linear:
            for i in range(batch_size):
                s1, s2 = i, (i + 1) % batch_size
                t1, t2 = s1 + batch_size, s2 + batch_size
                index_matrix[s1, s2] = 1. / float(batch_size)
                index_matrix[t1, t2] = 1. / float(batch_size)
                index_matrix[s1, t2] = -1. / float(batch_size)
                index_matrix[s2, t1] = -1. / float(batch_size)
        else:
            for i in range(batch_size):
                for j in range(batch_size):
                    if i != j:
                        index_matrix[i][j] = 1. / float(batch_size * (batch_size - 1))
                        index_matrix[i + batch_size][j + batch_size] = 1. / float(batch_size * (batch_size - 1))
            for i in range(batch_size):
                for j in range(batch_size):
                    index_matrix[i][j + batch_size] = -1. / float(batch_size * batch_size)
                    index_matrix[i + batch_size][j] = -1. / float(batch_size * batch_size)
    return index_matrix


class GaussianKernel(nn.Module):
    r"""Gaussian Kernel Matrix
    Args:
        sigma (float, optional): bandwidth :math:`\sigma`. Default: None
        track_running_stats (bool, optional): If ``True``, this module tracks the running mean of :math:`\sigma^2`.
          Otherwise, it won't track such statistics and always uses fix :math:`\sigma^2`. Default: ``True``
        alpha (float, optional): :math:`\alpha` which decides the magnitude of :math:`\sigma^2` when track_running_stats is set to ``True``

    Inputs:
        - X (tensor): input group :math:`X`

    Shape:
        - Inputs: :math:`(minibatch, F)` where F means the dimension of input features.
        - Outputs: :math:`(minibatch, minibatch)`
    """

    def __init__(self, sigma: Optional[float] = None, track_running_stats: Optional[bool] = True,
                 alpha: Optional[float] = 1.):
        super(GaussianKernel, self).__init__()
        assert track_running_stats or sigma is not None
        self.sigma_square = tr.tensor(sigma * sigma) if sigma is not None else None
        self.track_running_stats = track_running_stats
        self.alpha = alpha

    def forward(self, X: tr.Tensor) -> tr.Tensor:
        l2_distance_square = ((X.unsqueeze(0) - X.unsqueeze(1)) ** 2).sum(2)

        if self.track_running_stats:
            self.sigma_square = self.alpha * tr.mean(l2_distance_square.detach())

        return tr.exp(-l2_distance_square / (2 * self.sigma_square))


# =============================================================CDANE Function===========================================
def CDANE(input_list, ad_net, entropy=None, coeff=None, args=None, random_layer=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is None:
        op_out = tr.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))
    batch_size = softmax_output.size(0) // 2
    dc_target = tr.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float()
    if args.data_env != 'local':
        dc_target = dc_target.cuda()
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0 + tr.exp(-entropy)
        source_mask = tr.ones_like(entropy)
        source_mask[feature.size(0) // 2:] = 0
        source_weight = entropy * source_mask
        target_mask = tr.ones_like(entropy)
        target_mask[0:feature.size(0) // 2] = 0
        target_weight = entropy * target_mask
        weight = source_weight / tr.sum(source_weight).detach().item() + \
                 target_weight / tr.sum(target_weight).detach().item()
        return tr.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / tr.sum(
            weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target)


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()

    return fun1


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha=1):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024, use_cuda=True):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        if use_cuda:
            self.random_matrix = [tr.randn(input_dim_list[i], output_dim).cuda() for i in range(self.input_num)]
        else:
            self.random_matrix = [tr.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [tr.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = tr.mul(return_tensor, single)
        return return_tensor


# =============================================================MCC Function=============================================
class ClassConfusionLoss(nn.Module):
    """
    The class confusion loss

    Parameters:
        - **t** Optional(float): the temperature factor used in MCC
    """

    def __init__(self, t):
        super(ClassConfusionLoss, self).__init__()
        self.t = t

    def forward(self, output: tr.Tensor) -> tr.Tensor:
        n_sample, n_class = output.shape
        softmax_out = nn.Softmax(dim=1)(output / self.t)
        entropy_weight = Entropy(softmax_out).detach()
        entropy_weight = 1 + tr.exp(-entropy_weight)
        entropy_weight = (n_sample * entropy_weight / tr.sum(entropy_weight)).unsqueeze(dim=1)
        class_confusion_matrix = tr.mm((softmax_out * entropy_weight).transpose(1, 0), softmax_out)
        class_confusion_matrix = class_confusion_matrix / tr.sum(class_confusion_matrix, dim=1)
        mcc_loss = (tr.sum(class_confusion_matrix) - tr.trace(class_confusion_matrix)) / n_class
        return mcc_loss


# =============================================================MDD Function=============================================
class MarginDisparityDiscrepancy(nn.Module):
    r"""The margin disparity discrepancy (MDD) proposed in `Bridging Theory and Algorithm for Domain Adaptation (ICML 2019) <https://arxiv.org/abs/1904.05801>`_.

    MDD can measure the distribution discrepancy in domain adaptation.

    The :math:`y^s` and :math:`y^t` are logits output by the main head on the source and target domain respectively.
    The :math:`y_{adv}^s` and :math:`y_{adv}^t` are logits output by the adversarial head.

    The definition can be described as:

    .. math::
        \mathcal{D}_{\gamma}(\hat{\mathcal{S}}, \hat{\mathcal{T}}) =
        -\gamma \mathbb{E}_{y^s, y_{adv}^s \sim\hat{\mathcal{S}}} L_s (y^s, y_{adv}^s) +
        \mathbb{E}_{y^t, y_{adv}^t \sim\hat{\mathcal{T}}} L_t (y^t, y_{adv}^t),

    where :math:`\gamma` is a margin hyper-parameter, :math:`L_s` refers to the disparity function defined on the source domain
    and :math:`L_t` refers to the disparity function defined on the target domain.

    Args:
        source_disparity (callable): The disparity function defined on the source domain, :math:`L_s`.
        target_disparity (callable): The disparity function defined on the target domain, :math:`L_t`.
        margin (float): margin :math:`\gamma`. Default: 4
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``

    Inputs:
        - y_s: output :math:`y^s` by the main head on the source domain
        - y_s_adv: output :math:`y^s` by the adversarial head on the source domain
        - y_t: output :math:`y^t` by the main head on the target domain
        - y_t_adv: output :math:`y_{adv}^t` by the adversarial head on the target domain
        - w_s (optional): instance weights for source domain
        - w_t (optional): instance weights for target domain

    """

    def __init__(self, source_disparity: Callable, target_disparity: Callable,
                 margin: Optional[float] = 4, reduction: Optional[str] = 'mean'):
        super(MarginDisparityDiscrepancy, self).__init__()
        self.margin = margin
        self.reduction = reduction
        self.source_disparity = source_disparity
        self.target_disparity = target_disparity

    def forward(self, y_s: torch.Tensor, y_s_adv: torch.Tensor, y_t: torch.Tensor, y_t_adv: torch.Tensor,
                w_s: Optional[torch.Tensor] = None, w_t: Optional[torch.Tensor] = None) -> torch.Tensor:

        source_loss = -self.margin * self.source_disparity(y_s, y_s_adv)
        target_loss = self.target_disparity(y_t, y_t_adv)
        if w_s is None:
            w_s = torch.ones_like(source_loss)
        source_loss = source_loss * w_s
        if w_t is None:
            w_t = torch.ones_like(target_loss)
        target_loss = target_loss * w_t

        loss = source_loss + target_loss
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class ClassificationMarginDisparityDiscrepancy(MarginDisparityDiscrepancy):
    r"""
    The margin disparity discrepancy (MDD) proposed in `Bridging Theory and Algorithm for Domain Adaptation (ICML 2019) <https://arxiv.org/abs/1904.05801>`_.

    It measures the distribution discrepancy in domain adaptation
    for classification.

    When margin is equal to 1, it's also called disparity discrepancy (DD).

    The :math:`y^s` and :math:`y^t` are logits output by the main classifier on the source and target domain respectively.
    The :math:`y_{adv}^s` and :math:`y_{adv}^t` are logits output by the adversarial classifier.
    They are expected to contain raw, unnormalized scores for each class.

    The definition can be described as:

    .. math::
        \mathcal{D}_{\gamma}(\hat{\mathcal{S}}, \hat{\mathcal{T}}) =
        \gamma \mathbb{E}_{y^s, y_{adv}^s \sim\hat{\mathcal{S}}} \log\left(\frac{\exp(y_{adv}^s[h_{y^s}])}{\sum_j \exp(y_{adv}^s[j])}\right) +
        \mathbb{E}_{y^t, y_{adv}^t \sim\hat{\mathcal{T}}} \log\left(1-\frac{\exp(y_{adv}^t[h_{y^t}])}{\sum_j \exp(y_{adv}^t[j])}\right),

    where :math:`\gamma` is a margin hyper-parameter and :math:`h_y` refers to the predicted label when the logits output is :math:`y`.
    You can see more details in `Bridging Theory and Algorithm for Domain Adaptation <https://arxiv.org/abs/1904.05801>`_.

    Args:
        margin (float): margin :math:`\gamma`. Default: 4
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``

    Inputs:
        - y_s: logits output :math:`y^s` by the main classifier on the source domain
        - y_s_adv: logits output :math:`y^s` by the adversarial classifier on the source domain
        - y_t: logits output :math:`y^t` by the main classifier on the target domain
        - y_t_adv: logits output :math:`y_{adv}^t` by the adversarial classifier on the target domain

    Shape:
        - Inputs: :math:`(minibatch, C)` where C = number of classes, or :math:`(minibatch, C, d_1, d_2, ..., d_K)`
          with :math:`K \geq 1` in the case of `K`-dimensional loss.
        - Output: scalar. If :attr:`reduction` is ``'none'``, then the same size as the target: :math:`(minibatch)`, or
          :math:`(minibatch, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of K-dimensional loss.

    Examples::

        >>> num_classes = 2
        >>> batch_size = 10
        >>> loss = ClassificationMarginDisparityDiscrepancy(margin=4.)
        >>> # logits output from source domain and target domain
        >>> y_s, y_t = torch.randn(batch_size, num_classes), torch.randn(batch_size, num_classes)
        >>> # adversarial logits output from source domain and target domain
        >>> y_s_adv, y_t_adv = torch.randn(batch_size, num_classes), torch.randn(batch_size, num_classes)
        >>> output = loss(y_s, y_s_adv, y_t, y_t_adv)
    """

    def __init__(self, margin: Optional[float] = 4, **kwargs):
        def source_discrepancy(y: torch.Tensor, y_adv: torch.Tensor):
            _, prediction = y.max(dim=1)
            return F.cross_entropy(y_adv, prediction, reduction='none')

        def target_discrepancy(y: torch.Tensor, y_adv: torch.Tensor):
            _, prediction = y.max(dim=1)
            return -F.nll_loss(shift_log(1. - F.softmax(y_adv, dim=1)), prediction, reduction='none')

        super(ClassificationMarginDisparityDiscrepancy, self).__init__(source_discrepancy, target_discrepancy, margin,
                                                                       **kwargs)


def shift_log(x: torch.Tensor, offset: Optional[float] = 1e-6) -> torch.Tensor:
    r"""
    First shift, then calculate log, which can be described as:

    .. math::
        y = \max(\log(x+\text{offset}), 0)

    Used to avoid the gradient explosion problem in log(x) function when x=0.

    Args:
        x (torch.Tensor): input tensor
        offset (float, optional): offset size. Default: 1e-6

    .. note::
        Input tensor falls in [0., 1.] and the output tensor falls in [-log(offset), 0]
    """
    return torch.log(torch.clamp(x + offset, max=1.))


from typing import Optional, Any, Tuple

class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)


class WarmStartGradientReverseLayer(nn.Module):
    """Gradient Reverse Layer :math:`\mathcal{R}(x)` with warm start

        The forward and backward behaviours are:

        .. math::
            \mathcal{R}(x) = x,

            \dfrac{ d\mathcal{R}} {dx} = - \lambda I.

        :math:`\lambda` is initiated at :math:`lo` and is gradually changed to :math:`hi` using the following schedule:

        .. math::
            \lambda = \dfrac{2(hi-lo)}{1+\exp(- α \dfrac{i}{N})} - (hi-lo) + lo

        where :math:`i` is the iteration step.

        Args:
            alpha (float, optional): :math:`α`. Default: 1.0
            lo (float, optional): Initial value of :math:`\lambda`. Default: 0.0
            hi (float, optional): Final value of :math:`\lambda`. Default: 1.0
            max_iters (int, optional): :math:`N`. Default: 1000
            auto_step (bool, optional): If True, increase :math:`i` each time `forward` is called.
              Otherwise use function `step` to increase :math:`i`. Default: False
        """

    def __init__(self, alpha: Optional[float] = 1.0, lo: Optional[float] = 0.0, hi: Optional[float] = 1.,
                 max_iters: Optional[int] = 1000., auto_step: Optional[bool] = False):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """"""
        coeff = float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1


class GeneralModule(nn.Module):
    def __init__(self, backbone_dim: int, num_classes: int, bottleneck: nn.Module,
                 head: nn.Module, adv_head: nn.Module, grl: Optional[WarmStartGradientReverseLayer] = None,
                 finetune: Optional[bool] = True):
        super(GeneralModule, self).__init__()
        self.backbone_dim = backbone_dim
        self.num_classes = num_classes
        self.bottleneck = bottleneck
        self.head = head
        self.adv_head = adv_head
        self.finetune = finetune
        self.grl_layer = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000,
                                                       auto_step=False) if grl is None else grl

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """"""
        # removed feature extraction part
        features = self.bottleneck(x)
        outputs = self.head(features)
        features_adv = self.grl_layer(features)
        outputs_adv = self.adv_head(features_adv)
        if self.training:
            return outputs, outputs_adv
        else:
            return outputs

    def step(self):
        """
        Gradually increase :math:`\lambda` in GRL layer.
        """
        self.grl_layer.step()

    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        """
        Return a parameters list which decides optimization hyper-parameters,
        such as the relative learning rate of each layer.
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr if self.finetune else base_lr},
            {"params": self.bottleneck.parameters(), "lr": base_lr},
            {"params": self.head.parameters(), "lr": base_lr},
            {"params": self.adv_head.parameters(), "lr": base_lr}
        ]
        return params


class MDDClassifier(GeneralModule):
    r"""Classifier for MDD.

    Classifier for MDD has one backbone, one bottleneck, while two classifier heads.
    The first classifier head is used for final predictions.
    The adversarial classifier head is only used when calculating MarginDisparityDiscrepancy.


    Args:
        backbone (torch.nn.Module): Any backbone to extract 1-d features from data
        num_classes (int): Number of classes
        bottleneck_dim (int, optional): Feature dimension of the bottleneck layer. Default: 1024
        width (int, optional): Feature dimension of the classifier head. Default: 1024
        grl (nn.Module): Gradient reverse layer. Will use default parameters if None. Default: None.
        finetune (bool, optional): Whether use 10x smaller learning rate in the backbone. Default: True

    Inputs:
        - x (tensor): input data

    Outputs:
        - outputs: logits outputs by the main classifier
        - outputs_adv: logits outputs by the adversarial classifier

    Shape:
        - x: :math:`(minibatch, *)`, same shape as the input of the `backbone`.
        - outputs, outputs_adv: :math:`(minibatch, C)`, where C means the number of classes.

    .. note::
        Remember to call function `step()` after function `forward()` **during training phase**! For instance,

            >>> # x is inputs, classifier is an ImageClassifier
            >>> outputs, outputs_adv = classifier(x)
            >>> classifier.step()

    """

    def __init__(self, backbone_dim: int, num_classes: int,
                 bottleneck_dim: Optional[int] = 1024, width: Optional[int] = 1024,
                 grl: Optional[WarmStartGradientReverseLayer] = None, finetune=True, pool_layer=None):
        grl_layer = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000,
                                                       auto_step=False) if grl is None else grl
        # when not using feature extraction module, no need to pool
        '''
        if pool_layer is None:
            pool_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
        '''
        bottleneck = nn.Sequential(
            #pool_layer,
            nn.Linear(backbone_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        bottleneck[1].weight.data.normal_(0, 0.005)
        bottleneck[1].bias.data.fill_(0.1)

        # removed extra layers
        # The classifier head used for final predictions.
        head = nn.Sequential(
            nn.Linear(bottleneck_dim, num_classes)
        )
        # The adversarial classifier head
        adv_head = nn.Sequential(
            nn.Linear(bottleneck_dim, num_classes)
        )
        '''
        for dep in range(2):
            head[dep * 3].weight.data.normal_(0, 0.01)
            head[dep * 3].bias.data.fill_(0.0)
            adv_head[dep * 3].weight.data.normal_(0, 0.01)
            adv_head[dep * 3].bias.data.fill_(0.0)
        '''
        super(MDDClassifier, self).__init__(backbone_dim, num_classes, bottleneck,
                                              head, adv_head, grl_layer, finetune)


# =============================================================DSAN Function============================================
def lmmd(source, target, s_label, t_label, class_num, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = source.size()[0]
    weight_ss, weight_tt, weight_st = cal_weight(s_label, t_label, class_num=class_num)
    weight_ss = tr.from_numpy(weight_ss).cuda()
    weight_tt = tr.from_numpy(weight_tt).cuda()
    weight_st = tr.from_numpy(weight_st).cuda()

    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = tr.Tensor([0]).cuda()
    if tr.sum(tr.isnan(sum(kernels))):
        return loss
    SS = kernels[:batch_size, :batch_size]
    TT = kernels[batch_size:, batch_size:]
    ST = kernels[:batch_size, batch_size:]

    loss += tr.sum(weight_ss * SS + weight_tt * TT - 2 * weight_st * ST)
    loss = loss / batch_size  # calculate the mean
    return loss


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = tr.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = tr.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [tr.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


def convert_to_onehot(sca_label, class_num=2):

    return np.eye(class_num)[sca_label]


def cal_weight(s_label, t_label, class_num=None):
    batch_size = s_label.size()[0]
    s_sca_label = s_label.cpu().data.numpy()
    s_vec_label = convert_to_onehot(s_sca_label, class_num=class_num)
    s_sum = np.sum(s_vec_label, axis=0).reshape(1, class_num)
    s_sum[s_sum == 0] = 100
    s_vec_label = s_vec_label / s_sum

    # use prediction probability pseudo label in UDA
    #t_sca_label = t_label.cpu().data.max(1)[1].numpy()
    #t_vec_label = t_label.cpu().data.numpy()

    # use hard true label in SDA
    t_sca_label = t_label.cpu().data.numpy()
    t_vec_label = convert_to_onehot(t_sca_label, class_num=class_num)

    t_sum = np.sum(t_vec_label, axis=0).reshape(1, class_num)
    t_sum[t_sum == 0] = 100
    t_vec_label = t_vec_label / t_sum

    weight_ss = np.zeros((batch_size, batch_size))
    weight_tt = np.zeros((batch_size, batch_size))
    weight_st = np.zeros((batch_size, batch_size))

    set_s = set(s_sca_label)
    set_t = set(t_sca_label)
    count = 0
    for i in range(class_num):
        if i in set_s and i in set_t:
            s_tvec = s_vec_label[:, i].reshape(batch_size, -1)
            t_tvec = t_vec_label[:, i].reshape(batch_size, -1)
            ss = np.dot(s_tvec, s_tvec.T)
            weight_ss = weight_ss + ss# / np.sum(s_tvec) / np.sum(s_tvec)
            tt = np.dot(t_tvec, t_tvec.T)
            weight_tt = weight_tt + tt# / np.sum(t_tvec) / np.sum(t_tvec)
            st = np.dot(s_tvec, t_tvec.T)
            weight_st = weight_st + st# / np.sum(s_tvec) / np.sum(t_tvec)
            count += 1

    length = count  # len( set_s ) * len( set_t )
    if length != 0:
        weight_ss = weight_ss / length
        weight_tt = weight_tt / length
        weight_st = weight_st / length
    else:
        weight_ss = np.array([0])
        weight_tt = np.array([0])
        weight_st = np.array([0])
    return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')


# =============================================================MSFAN Function===========================================
def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = tr.mean(XX + YY - XY -YX)
    return loss


# =============================================================JAN Function=============================================
class JointMultipleKernelMaximumMeanDiscrepancy(nn.Module):
    r"""The Joint Multiple Kernel Maximum Mean Discrepancy (JMMD) used in
    `Deep Transfer Learning with Joint Adaptation Networks (ICML 2017) <https://arxiv.org/abs/1605.06636>`_
    Given source domain :math:`\mathcal{D}_s` of :math:`n_s` labeled points and target domain :math:`\mathcal{D}_t`
    of :math:`n_t` unlabeled points drawn i.i.d. from P and Q respectively, the deep networks will generate
    activations in layers :math:`\mathcal{L}` as :math:`\{(z_i^{s1}, ..., z_i^{s|\mathcal{L}|})\}_{i=1}^{n_s}` and
    :math:`\{(z_i^{t1}, ..., z_i^{t|\mathcal{L}|})\}_{i=1}^{n_t}`. The empirical estimate of
    :math:`\hat{D}_{\mathcal{L}}(P, Q)` is computed as the squared distance between the empirical kernel mean
    embeddings as
    .. math::
        \hat{D}_{\mathcal{L}}(P, Q) &=
        \dfrac{1}{n_s^2} \sum_{i=1}^{n_s}\sum_{j=1}^{n_s} \prod_{l\in\mathcal{L}} k^l(z_i^{sl}, z_j^{sl}) \\
        &+ \dfrac{1}{n_t^2} \sum_{i=1}^{n_t}\sum_{j=1}^{n_t} \prod_{l\in\mathcal{L}} k^l(z_i^{tl}, z_j^{tl}) \\
        &- \dfrac{2}{n_s n_t} \sum_{i=1}^{n_s}\sum_{j=1}^{n_t} \prod_{l\in\mathcal{L}} k^l(z_i^{sl}, z_j^{tl}). \\
    Args:
        kernels (tuple(tuple(torch.nn.Module))): kernel functions, where `kernels[r]` corresponds to kernel :math:`k^{\mathcal{L}[r]}`.
        linear (bool): whether use the linear version of JAN. Default: False
        thetas (list(Theta): use adversarial version JAN if not None. Default: None
    Inputs:
        - z_s (tuple(tensor)): multiple layers' activations from the source domain, :math:`z^s`
        - z_t (tuple(tensor)): multiple layers' activations from the target domain, :math:`z^t`
    Shape:
        - :math:`z^{sl}` and :math:`z^{tl}`: :math:`(minibatch, *)`  where * means any dimension
        - Outputs: scalar
    .. note::
        Activations :math:`z^{sl}` and :math:`z^{tl}` must have the same shape.
    .. note::
        The kernel values will add up when there are multiple kernels for a certain layer.
    """

    def __init__(self, kernels: Sequence[Sequence[nn.Module]], linear: Optional[bool] = True, thetas: Sequence[nn.Module] = None):
        super(JointMultipleKernelMaximumMeanDiscrepancy, self).__init__()
        self.kernels = kernels
        self.index_matrix = None
        self.linear = linear
        if thetas:
            self.thetas = thetas
        else:
            self.thetas = [nn.Identity() for _ in kernels]

    def forward(self, z_s: tr.Tensor, z_t: tr.Tensor) -> tr.Tensor:
        batch_size = int(z_s[0].size(0))
        self.index_matrix = _update_index_matrix(batch_size, self.index_matrix, self.linear).to(z_s[0].device)

        kernel_matrix = tr.ones_like(self.index_matrix)
        for layer_z_s, layer_z_t, layer_kernels, theta in zip(z_s, z_t, self.kernels, self.thetas):
            layer_features = tr.cat([layer_z_s, layer_z_t], dim=0)
            layer_features = theta(layer_features)
            kernel_matrix *= sum(
                [kernel(layer_features) for kernel in layer_kernels])  # Add up the matrix of each kernel

        # Add 2 / (n-1) to make up for the value on the diagonal
        # to ensure loss is positive in the non-linear version
        loss = (kernel_matrix * self.index_matrix).sum() + 2. / float(batch_size - 1)
        return loss


# ============================================================MSTN Function=============================================
class MSTN(nn.Module):
    def __init__(self, n_class, deep_feature_dim, cuda, euclidean_distance=False):
        super(MSTN, self).__init__()
        self.n_class = n_class
        self.deep_feature_dim = deep_feature_dim
        self.cuda = cuda
        self.decay = 0.3  # hyperparameter
        self.s_centroid = torch.zeros(self.n_class, self.deep_feature_dim)
        self.t_centroid = torch.zeros(self.n_class, self.deep_feature_dim)
        if self.cuda:
            self.s_centroid = self.s_centroid.cuda()
            self.t_centroid = self.t_centroid.cuda()
        self.init()

        self.euclidean_distance = euclidean_distance

    def init(self):
        self.CEloss, self.MSEloss, self.BCEloss = nn.CrossEntropyLoss(), nn.MSELoss(), nn.BCEWithLogitsLoss(reduction='mean')
        if self.cuda:
           self.CEloss, self.MSEloss, self.BCEloss = self.CEloss.cuda(), self.MSEloss.cuda(), self.BCEloss.cuda()

    def adloss(self, s_logits, t_logits, s_feature, t_feature, y_s, y_t):
        n, d = s_feature.shape

        # get labels using pseudo labels for target in UDA
        #s_labels, t_labels = y_s, torch.max(y_t, 1)[1]

        # use hard labels in SDA
        s_labels, t_labels = y_s, y_t

        # image number in each class
        ones = torch.ones_like(s_labels, dtype=torch.float)
        zeros = torch.zeros(self.n_class)
        if self.cuda:
            zeros = zeros.cuda()
        s_n_classes = zeros.scatter_add(0, s_labels, ones)
        t_n_classes = zeros.scatter_add(0, t_labels, ones)

        # image number cannot be 0, when calculating centroids
        ones = torch.ones_like(s_n_classes)
        s_n_classes = torch.max(s_n_classes, ones)
        t_n_classes = torch.max(t_n_classes, ones)

        # calculating centroids, sum and divide
        zeros = torch.zeros(self.n_class, d)
        if self.cuda:
            zeros = zeros.cuda()
        s_sum_feature = zeros.scatter_add(0, torch.transpose(s_labels.repeat(d, 1), 1, 0), s_feature)
        t_sum_feature = zeros.scatter_add(0, torch.transpose(t_labels.repeat(d, 1), 1, 0), t_feature)
        current_s_centroid = torch.div(s_sum_feature, s_n_classes.view(self.n_class, 1))
        current_t_centroid = torch.div(t_sum_feature, t_n_classes.view(self.n_class, 1))

        # Moving Centroid
        decay = self.decay
        s_centroid = (1 - decay) * self.s_centroid + decay * current_s_centroid
        t_centroid = (1 - decay) * self.t_centroid + decay * current_t_centroid


        if self.euclidean_distance:
            center_dists = 0
            for i in range(self.n_class):
                center_dists += torch.cdist(s_centroid[i].reshape(1, -1), t_centroid[i].reshape(1, -1))
            semantic_loss = center_dists / self.n_class
        else:
            semantic_loss = self.MSEloss(s_centroid, t_centroid)

        self.s_centroid = s_centroid.detach()
        self.t_centroid = t_centroid.detach()

        '''
        # sigmoid binary cross entropy with reduce mean
        D_real_loss = self.BCEloss(t_logits, torch.ones_like(t_logits))
        D_fake_loss = self.BCEloss(s_logits, torch.zeros_like(s_logits))
        D_loss = (D_real_loss + D_fake_loss) * 0.1
        G_loss = -D_loss
        return G_loss, D_loss, semantic_loss
        '''

        return None, None, semantic_loss


# ============================================================CCSA Function=============================================
def csa(source, target, class_eq):
    margin = 1
    dist = F.pairwise_distance(source, target)
    loss = class_eq * dist.pow(2)
    loss += (1 - class_eq) * (margin - dist).clamp(min=0).pow(2)
    return loss.mean()


# ============================================================ATM Function==============================================
def atm_loss(features, labels, num_classes, source_num=None, left_weight=1, right_weight=1, cross_weight=1, inter_weight=0):
    # left weight is for source intra class
    # right weight is for target intra class
    # cross weight is for cross-domain distance
    # inter weight is for target inter class
    softmax_out = nn.Softmax(dim=1)(features)
    batch_size = features.size(0)
    if float(batch_size) % 2 != 0:
        raise Exception('Incorrect batch size provided')

    if source_num:
        batch_left = softmax_out[:source_num]
        batch_right = softmax_out[source_num:]
        labels_left = labels[:source_num]
        labels_right = labels[source_num:]
    else:
        batch_left = softmax_out[:int(0.5 * batch_size)]
        batch_right = softmax_out[int(0.5 * batch_size):]
        labels_left = labels[:int(0.5 * batch_size)]
        labels_right = labels[int(0.5 * batch_size):]

    if cross_weight != 0:
        cross_loss = torch.norm((batch_left - batch_right).abs(), 2, 1).sum() / float(batch_size)
    else:
        cross_loss = 0
    batch_left_loss = get_pari_loss1(labels_left, batch_left)
    batch_right_loss = get_pari_loss1(labels_right, batch_right)

    # check all classes exist
    if len(torch.unique(labels_right)) == num_classes:
        inter_loss = get_pari_loss2(labels_right, batch_right)
    else:
        inter_loss = 0

    return cross_weight * cross_loss + left_weight * batch_left_loss + right_weight * batch_right_loss - inter_weight * inter_loss


def get_pari_loss1(labels, features):
    loss = 0
    count = 0
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if (labels[i] == labels[j]):
                count += 1
                loss += torch.norm((features[i] - features[j]).abs(), 2, 0).sum()
    return loss / count


def get_pari_loss2(labels, features):
    loss = 0
    count = 0
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if (labels[i] != labels[j]):
                count += 1
                loss += torch.norm((features[i] - features[j]).abs(), 2, 0).sum()
    return loss / count


# =========================================================centerloss Function==========================================
class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

# ============================================================ODL Function==============================================
def odl_loss(features, labels, num_classes, margin, args, orientation=False):
    # Orientational Distribution Learning With Hierarchical Spatial Attention for Open Set Recognition

    centers = []
    for i in range(num_classes):
        inds_class = torch.where(labels == i)[0]
        features_class = features[inds_class, :]
        centers.append(torch.mean(features_class, dim=0))
    centers = torch.stack(centers)

    com_loss = 0
    max_intra_dists = 0
    max_intra_dist = 0
    for i in range(num_classes):
        inds_class = torch.where(labels == i)[0]
        features_class = features[inds_class, :]
        #if squared:
        #    dists = torch.pow(torch.cdist(features_class, centers[i].reshape(1, -1)), 2)
        #else:
        dists = torch.cdist(features_class, centers[i].reshape(1, -1))
        #print('intra dists:', dists)
        #input('')

        dists_flatten = dists.reshape(-1,)
        for j in range(len(dists_flatten)):
            if dists_flatten[j] > max_intra_dist:
                max_intra_dist = dists_flatten[j]
        max_intra_dists += max_intra_dist

        sum_dist = torch.sum(dists)
        com_loss += sum_dist
    com_loss /= len(features)

    #intra_loss = com_loss
    if orientation:
        dis_loss = 0
        for i in range(len(features)):
            v_i = centers[labels[i]] - features[i]
            cosine_dist = []
            for j in range(num_classes):
                if j == labels[i]:
                    continue
                v_j = centers[j] - features[i]
                cosine_sim = nn.CosineSimilarity()(v_i.reshape(1, -1), v_j.reshape(1, -1))
                cosine_dist.append(1 - cosine_sim)
            cosine_dist = torch.concat(cosine_dist)
            #print('cosine_dist', cosine_dist)
            other_centers = centers[np.delete(np.arange(num_classes, dtype=int), labels[i].cpu())]
            center_dists = torch.cdist(features[i].reshape(1, -1), other_centers)
            w = torch.exp(-1 * center_dists) / torch.sum(torch.exp((-1 * center_dists)))
            w = w.reshape(-1, )
            #print('w', w)
            sample_loss = torch.dot(w, cosine_dist)
            #print('sample dis loss', sample_loss)
            dis_loss += sample_loss
        dis_loss /= len(features)
        #print('com dis', com_loss, dis_loss)
        intra_loss = com_loss + dis_loss
    else:
        intra_loss = com_loss
    #input('')

    max_intra_dists *= (2 / args.class_num)

    if margin == 'auto':
        args.m_margin = max_intra_dists.detach().item()
        print('setting margin to be:', max_intra_dists)
        margin = args.m_margin

    #if squared:
    #    center_dists = torch.pow(torch.cdist(centers, centers), 2)
    #else:
    center_dists = torch.cdist(centers, centers)
    #print('center_dists', center_dists)
    sep_loss = 0
    for i in range(num_classes):
        center_dist = center_dists[i]

        #print('center_dist', center_dist)

        # center of the only closest class
        min_val = 2 ** 10
        for j in range(len(center_dist)):
            if center_dist[j] != 0. and center_dist[j] < min_val:
                min_val = center_dist[j]
        #print('min_val', min_val)
        if args.m_margin is None:
            dist = min_val
        else:
            zero = torch.tensor(0.)
            if args.data_env != 'local':
                zero = zero.cuda()
            #print('min_val', min_val)
            dist = max(margin - min_val, zero)
            #print('dist', dist)
            #input('')
        sep_loss += dist
        #print('inter max of margin - min dist:', dist)

        '''
        # centers of all classes
        dist = []
        for j in range(len(center_dist)):
            if center_dist[j] != 0.:
                zero = torch.tensor(0.)
                if args.data_env != 'local':
                    zero = zero.cuda()
                dist.append(max(margin - center_dist[j], zero))
        dist = torch.mean(torch.stack(dist), dim=0)
        sep_loss += dist
        '''

    sep_loss /= num_classes
    #inter_loss = sep_loss

    '''
    ori_loss = 0
    if num_classes > 2:
        for i in range(num_classes):
            center_dist = center_dists[i]
            print('center_dist', center_dist)

            min_val = 2 ** 10
            min_ind = -1
            for j in range(len(center_dist)):
                if center_dist[j] != 0. and center_dist[j] < min_val:
                    min_val = center_dist[j]
                    min_ind = j
            print('min_ind', min_ind)

            v_s = centers[min_ind] - centers[i]

            cosine_dist = []
            for j in range(num_classes):
                if j == i or j == min_ind:
                    continue

                v_0 = centers[j] - centers[i]
                cosine_sim = nn.CosineSimilarity()(v_s.reshape(1, -1), v_0.reshape(1, -1))
                cosine_dist.append(1 + cosine_sim)
            cosine_dist = torch.concat(cosine_dist)
            #print('cosine_dist', cosine_dist)
            other_nonclosest_centers = centers[np.delete(np.arange(num_classes, dtype=int), [i, min_ind])]
            center_dists = torch.cdist(centers[min_ind].reshape(1, -1), other_nonclosest_centers)
            w = torch.exp(-1 * center_dists) / torch.sum(torch.exp((-1 * center_dists)))
            w = w.reshape(-1, )
            #print('w', w)
            class_loss = torch.dot(w, cosine_dist)
            #print('sample dis loss', sample_loss)
            ori_loss += class_loss
    ori_loss /= num_classes
    inter_loss = sep_loss + ori_loss
    '''
    inter_loss = sep_loss

    return intra_loss, inter_loss

# ============================================================TCL Function==============================================
def tcl_loss(features, labels, num_classes, margin, args, squared=False):
    # triplet center loss
    # Triplet-Center Loss for Multi-View 3D Object Retrieval

    centers = []
    for i in range(num_classes):
        inds_class = torch.where(labels == i)[0]
        features_class = features[inds_class, :]
        centers.append(torch.mean(features_class, dim=0))
    centers = torch.stack(centers)

    tcl_loss = 0
    for i in range(len(features)):
        this_center = centers[labels[i]]
        other_inds = np.delete(np.arange(num_classes, dtype=int), labels[i].cpu())
        other_centers = centers[other_inds]

        if squared:
            this_dist = torch.pow(torch.cdist(features[i].reshape(1, -1), this_center.reshape(1, -1)), 2)
            other_dists = torch.pow(torch.cdist(features[i].reshape(1, -1), other_centers), 2)
        else:
            this_dist = torch.cdist(features[i].reshape(1, -1), this_center.reshape(1, -1))
            other_dists = torch.cdist(features[i].reshape(1, -1), other_centers)
        other_dists = other_dists.reshape(-1, )

        # center of the only closest class
        min_val = 2 ** 10
        for j in range(len(other_dists)):
            if other_dists[j] < min_val:
                min_val = other_dists[j]

        zero = torch.tensor([[0.]])
        if args.data_env != 'local':
            zero = zero.cuda()
        dist = max(this_dist - min_val + margin, zero)[0, 0]
        tcl_loss += dist

    tcl_loss /= len(features)

    return tcl_loss

# ============================================================BSP Function==============================================
class BatchSpectralPenalizationLoss(nn.Module):
    r"""Batch spectral penalization loss from `Transferability vs. Discriminability: Batch
    Spectral Penalization for Adversarial Domain Adaptation (ICML 2019)
    <http://ise.thss.tsinghua.edu.cn/~mlong/doc/batch-spectral-penalization-icml19.pdf>`_.

    Given source features :math:`f_s` and target features :math:`f_t` in current mini batch, singular value
    decomposition is first performed

    .. math::
        f_s = U_s\Sigma_sV_s^T

    .. math::
        f_t = U_t\Sigma_tV_t^T

    Then batch spectral penalization loss is calculated as

    .. math::
        loss=\sum_{i=1}^k(\sigma_{s,i}^2+\sigma_{t,i}^2)

    where :math:`\sigma_{s,i},\sigma_{t,i}` refer to the :math:`i-th` largest singular value of source features
    and target features respectively. We empirically set :math:`k=1`.

    Inputs:
        - f_s (tensor): feature representations on source domain, :math:`f^s`
        - f_t (tensor): feature representations on target domain, :math:`f^t`

    Shape:
        - f_s, f_t: :math:`(N, F)` where F means the dimension of input features.
        - Outputs: scalar.

    """

    def __init__(self):
        super(BatchSpectralPenalizationLoss, self).__init__()

    def forward(self, f_s, f_t):
        _, s_s, _ = torch.svd(f_s)
        _, s_t, _ = torch.svd(f_t)
        loss = torch.pow(s_s[0], 2) + torch.pow(s_t[0], 2)
        return loss


# ============================================================SCL Function==============================================
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


# ============================================================MMD Function==============================================
class MMD(nn.Module):
    def __init__(self, gaussian):
        super(MMD, self).__init__()
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.1, 1, 10]):#gamma=[0.001, 0.01, 0.1, 1, 10, 100, 1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def forward(self, x, y):
        if self.kernel_type == "gaussian":
            # MMD using gaussian kerner
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            # MMD using mean and covariance difference
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

# ============================================================SVM Function==============================================
class MultiClassHingeLoss(nn.Module):
    def __init__(self, C=1.0):
        super(MultiClassHingeLoss, self).__init__()
        self.C = C

    def forward(self, outputs, labels):
        num_classes = outputs.size(1)
        correct_outputs = outputs[torch.arange(outputs.size(0)), labels].unsqueeze(1)
        margin = 1.0 + outputs - correct_outputs
        margin[torch.arange(margin.size(0)), labels] = 0
        hinge_loss = torch.max(margin, torch.zeros_like(margin)).mean()
        return hinge_loss
