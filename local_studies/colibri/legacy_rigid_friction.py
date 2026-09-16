"""Diagnostic legacy friction adapter; leaves shared solver files untouched."""
import warp as wp
from newton._src.solvers.phoenx.constraints.contact_container import ContactContainer
from newton._src.solvers.phoenx.constraints.contact_projection import contact_project_velocity_update_no_soft_pd

@wp.func
def legacy_update(
    cc: ContactContainer,
    k: wp.int32,
    normal: wp.vec3f,
    tangent1: wp.vec3f,
    tangent2: wp.vec3f,
    jv_n: wp.float32,
    jv_t1: wp.float32,
    jv_t2: wp.float32,
    eff_n: wp.float32,
    eff_t1: wp.float32,
    eff_t2: wp.float32,
    bias_n: wp.float32,
    bias_t1: wp.float32,
    bias_t2: wp.float32,
    mu_s: wp.float32,
    mu_k: wp.float32,
    mass_coeff_n: wp.float32,
    impulse_coeff_n: wp.float32,
    sor_boost: wp.float32,
    pd_eff_soft_n: wp.float32,
    pd_gamma_n: wp.float32,
    pd_bias_n: wp.float32,
    mobility_nt1: wp.float32,
    mobility_nt2: wp.float32,
    mobility_t1t2: wp.float32,
) -> wp.vec3f:
    return contact_project_velocity_update_no_soft_pd(cc, k, normal, tangent1, tangent2, jv_n, jv_t1, jv_t2, eff_n, eff_t1, eff_t2, bias_n, bias_t1, bias_t2, mu_s, mu_k, mass_coeff_n, impulse_coeff_n, sor_boost, pd_eff_soft_n, pd_gamma_n, pd_bias_n)

def install():
    from newton._src.solvers.phoenx.constraints import constraint_contact_cloth as c
    c.contact_project_coupled_velocity_update_no_soft_pd = legacy_update
