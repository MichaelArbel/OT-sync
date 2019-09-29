
import geoopt
import torch

EPS = {torch.float32: 1e-4, torch.float64: 1e-8}

_sphere_doc = r"""
    Sphere manifold induced by the following constraint
    .. math::
        \|x\|=1\\
        x \in \mathbb{span}(U)
    where :math:`U` can be parametrized with compliment space or intersection.
    Parameters
    ----------
    intersection : tensor
        shape ``(..., dim, K)``, subspace to intersect with
    complement : tensor
        shape ``(..., dim, K)``, subspace to compliment
"""


class HyperSphere(Manifold):
    
    ndim = 1
    name = "Sphere"
    reversible = False

    def __init__(self):
        super().__init__()
        
        self._configure_manifold_no_constraints()
        if (
            self.projector is not None
            and (geoopt.linalg.batch_linalg.matrix_rank(self.projector) == 1).any()
        ):
            raise ValueError(
                "Manifold only consists of isolated points when "
                "subspace is 1-dimensional."
            )

    def _check_shape(self, shape, name):
        ok, reason = super()._check_shape(shape, name)
        if ok and self.projector is not None:
            ok = len(shape) < (self.projector.dim() - 1)
            if not ok:
                reason = "`{}` should have at least {} dimensions but has {}".format(
                    name, self.projector.dim() - 1, len(shape)
                )
            ok = shape[-1] == self.projector.shape[-2]
            if not ok:
                reason = "The [-2] shape of `span` does not match `{}`: {}, {}".format(
                    name, shape[-1], self.projector.shape[-1]
                )
        elif ok:
            ok = shape[-1] != 1
            if not ok:
                reason = (
                    "Manifold only consists of isolated points when "
                    "subspace is 1-dimensional."
                )
        return ok, reason

    def _check_point_on_manifold(self, x, *, atol=1e-5, rtol=1e-5):
        norm = x.norm(dim=-1)
        ok = torch.allclose(norm, norm.new((1,)).fill_(1), atol=atol, rtol=rtol)
        if not ok:
            return False, "`norm(x) != 1` with atol={}, rtol={}".format(atol, rtol)
        ok = torch.allclose(self._project_on_subspace(x), x, atol=atol, rtol=rtol)
        if not ok:
            return (
                False,
                "`x` is not in the subspace of the manifold with atol={}, rtol={}".format(
                    atol, rtol
                ),
            )
        return True, None

    def _check_vector_on_tangent(self, x, u, *, atol=1e-5, rtol=1e-5):
        inner = self.inner(None, x, u, keepdim=True)
        ok = torch.allclose(inner, inner.new_zeros((1,)), atol=atol, rtol=rtol)
        if not ok:
            return False, "`<x, u> != 0` with atol={}, rtol={}".format(atol, rtol)
        return True, None

    def inner(self, x, u, v=None, *, keepdim=False):
        if v is None:
            v = u
        return (u * v).sum(-1, keepdim=keepdim)

    def projx(self, x):
        x = self._project_on_subspace(x)
        return x / x.norm(dim=-1, keepdim=True)

    def proju(self, x, u):
        u = u - (x * u).sum(dim=-1, keepdim=True) * x
        return self._project_on_subspace(u)

    def expmap(self, x, u):
        norm_u = u.norm(dim=-1, keepdim=True)
        exp = x * torch.cos(norm_u) + u * torch.sin(norm_u) / norm_u
        retr = self.projx(x + u)
        cond = norm_u > EPS[norm_u.dtype]
        return torch.where(cond, exp, retr)

    def retr(self, x, u):
        return self.projx(x + u)

    def transp_follow_retr(self, x, u, v, *more):
        y = self.retr(x, u)
        return self.transp(x, y, v, *more)

    def transp(self, x, y, v, *more):
        result = tuple(self.proju(y, _v) for _v in (v,) + more)
        return strip_tuple(result)

    def transp_follow_expmap(self, x, u, v, *more):
        y = self.expmap(x, u)
        return self.transp(x, y, v, *more)

    def expmap_transp(self, x, u, v, *more):
        y = self.expmap(x, u)
        vs = self.transp(x, y, v, *more)
        return (y,) + make_tuple(vs)

    def retr_transp(self, x, u, v, *more):
        y = self.retr(x, u)
        vs = self.transp(x, y, v, *more)
        return (y,) + make_tuple(vs)

    def logmap(self, x, y):
        u = self.proju(x, y - x)
        dist = self.dist(x, y, keepdim=True)
        # If the two points are "far apart", correct the norm.
        cond = dist.gt(EPS[dist.dtype])
        return torch.where(cond, u * dist / u.norm(dim=-1, keepdim=True), u)

    def dist(self, x, y, *, keepdim=False):
        inner = self.inner(None, x, y, keepdim=keepdim).clamp(-1, 1)
        return torch.acos(inner)

    egrad2rgrad = proju

    def _project_on_subspace(self, x):
        return x

    def random_uniform(self, *size, dtype=None, device=None):
        """
        Uniform random measure on Sphere manifold.
        Parameters
        ----------
        size : shape
            the desired output shape
        dtype : torch.dtype
            desired dtype
        device : torch.device
            desired device
        Returns
        -------
        ManifoldTensor
            random point on Sphere manifold
        Notes
        -----
        """
        self._assert_check_shape(size2shape(*size), "x")
        tens = torch.randn(*size, device=device, dtype=dtype)
        return ManifoldTensor(self.projx(tens), manifold=self)


class HyperSphereExact(HyperSphere):
    __doc__ = r"""{}
    See Also
    --------
    :class:`HyperSphere`
    
    Notes
    -----
    The implementation of retraction is an exact exponential map, this retraction will be used in optimization
    """.format(
        _sphere_doc
    )

    retr_transp = Sphere.expmap_transp
    transp_follow_retr = Sphere.transp_follow_expmap
    retr = Sphere.expmap

    def extra_repr(self):
        return "exact"