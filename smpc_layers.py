import torch
import logging
import syft as sy
import torch as th
import torch.nn as nn

# https://stackoverflow.com/questions/54421029/python-websockets-how-to-setup-connect-timeout
hook = sy.TorchHook(th)

me = hook.local_worker
me.is_client_worker = False

class SmpcLinearFunction(th.autograd.Function):
    workers = []
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(weight, bias)
        ctx._input = input

        # Performed in parallel in SA
        # SA also sends out the output to the individual nodes
        input = input.fix_prec().share(SmpcLinearFunction.workers[0], SmpcLinearFunction.workers[1])
        ctx.__input = input
        weight = weight.fix_prec().share(SmpcLinearFunction.workers[0], SmpcLinearFunction.workers[1])

        output = input.mm(weight.t()).get().float_precision().clone().detach()

        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output.clone()

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        weight, bias = ctx.saved_tensors
        input = ctx._input
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            # Performed in parallel in SA
            # SA also sends out the output to the individual nodes
            _grad_output = grad_output.fix_prec().share(SmpcLinearFunction.workers[0], SmpcLinearFunction.workers[1])

            grad_weight = _grad_output.t().mm(ctx.__input).get().float_precision().clone().detach()

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias

class SmpcLinearCut(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        # Not a very smart way to initialize weights
        self.weight.data.uniform_(-0.1, 0.1)
        if self.bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)

        SmpcLinearFunction.workers = [ sy.VirtualWorker(id=f"Worker{i}", hook=hook, is_client_worker=False) for i in range(2) ]
        # TODO ulimit -Sn 65536

    def forward(self, input):
        out = SmpcLinearFunction.apply(input, self.weight, self.bias)
        # See the autograd section for explanation of what happens here.
        return out
