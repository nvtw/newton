// Native CUDA snippets used by example_pyoptix_empty_buffer.py.
// Sections are loaded by name and string-joined into generated CUDA source.

// -- section: common_header_notrace -- begin
static __forceinline__ __device__ unsigned int optixGetLaunchIndexX()
{
    unsigned int u0;
    asm("call (%0), _optix_get_launch_index_x, ();" : "=r"(u0) :);
    return u0;
}

static __forceinline__ __device__ unsigned int optixGetLaunchIndexY()
{
    unsigned int u1;
    asm("call (%0), _optix_get_launch_index_y, ();" : "=r"(u1) :);
    return u1;
}

static __forceinline__ __device__ void optixSetPayload0(unsigned int p)
{
    asm volatile("call _optix_set_payload, (%0, %1);" : : "r"(0), "r"(p) :);
}
// -- section: common_header_notrace -- end

// -- section: launch_params -- begin
struct LaunchParams {
    unsigned int* image;
    unsigned int width;
    unsigned int height;
    float time_sec;
    unsigned long long trav_handle;
};

extern "C" {
__constant__ LaunchParams params;
}
// -- section: launch_params -- end

// -- section: trace_mode_notrace -- begin
extern "C" __global__ void __raygen__cube()
{
    const unsigned int ix = optixGetLaunchIndexX();
    const unsigned int iy = optixGetLaunchIndexY();
    if (ix >= params.width || iy >= params.height)
        return;

    const float t = params.time_sec;
    const float u = ((float)ix + 0.5f) / (float)params.width;
    const float v = ((float)iy + 0.5f) / (float)params.height;
    const float r = 0.5f + 0.5f * __sinf(6.2831853f * (u + 0.20f * t));
    const float g = 0.5f + 0.5f * __sinf(6.2831853f * (v + 0.35f * t));
    const float b = 0.5f + 0.5f * __sinf(6.2831853f * (u + v + 0.15f * t));
    const unsigned int ri = (unsigned int)(255.0f * r);
    const unsigned int gi = (unsigned int)(255.0f * g);
    const unsigned int bi = (unsigned int)(255.0f * b);
    params.image[iy * params.width + ix] = (255u << 24) | (bi << 16) | (gi << 8) | ri;
}

extern "C" __global__ void __miss__cube() { optixSetPayload_0(0xFF201020u); }

extern "C" __global__ void __closesthit__cube() { optixSetPayload_0(0xFF20A0F0u); }
// -- section: trace_mode_notrace -- end

// -- section: trace_mode_trace0 -- begin
extern "C" __global__ void __raygen__cube()
{
    const uint3 idx = optixGetLaunchIndex();
    const unsigned int ix = idx.x;
    const unsigned int iy = idx.y;
    if (ix >= params.width || iy >= params.height)
        return;

    const float fx = (2.0f * ((float)ix + 0.5f) / (float)params.width - 1.0f);
    const float fy = -(2.0f * ((float)iy + 0.5f) / (float)params.height - 1.0f);
    const float aspect = (float)params.width / (float)params.height;

    const float t = params.time_sec;
    const float cam_x = 2.2f * __sinf(0.45f * t);
    const float cam_y = 0.9f;
    const float cam_z = 2.2f * __cosf(0.45f * t);

    // Camera basis looking at world origin.
    float fwd_x = -cam_x;
    float fwd_y = -cam_y;
    float fwd_z = -cam_z;
    float fwd_inv_len = rsqrtf(fwd_x * fwd_x + fwd_y * fwd_y + fwd_z * fwd_z);
    fwd_x *= fwd_inv_len;
    fwd_y *= fwd_inv_len;
    fwd_z *= fwd_inv_len;

    float right_x = -fwd_z;
    float right_y = 0.0f;
    float right_z = fwd_x;
    float right_inv_len = rsqrtf(right_x * right_x + right_z * right_z + 1.0e-8f);
    right_x *= right_inv_len;
    right_z *= right_inv_len;

    float up_x = right_y * fwd_z - right_z * fwd_y;
    float up_y = right_z * fwd_x - right_x * fwd_z;
    float up_z = right_x * fwd_y - right_y * fwd_x;

    float dx = fx * aspect * right_x + fy * up_x + 1.8f * fwd_x;
    float dy = fx * aspect * right_y + fy * up_y + 1.8f * fwd_y;
    float dz = fx * aspect * right_z + fy * up_z + 1.8f * fwd_z;
    const float inv_len = rsqrtf(dx * dx + dy * dy + dz * dz);
    dx *= inv_len;
    dy *= inv_len;
    dz *= inv_len;

    unsigned int payload0 = 0xFF102030u;
    optixTrace(
        params.trav_handle, make_float3(cam_x, cam_y, cam_z), make_float3(dx, dy, dz), 0.001f, 1.0e16f, 0.0f, 255u, 0u,
        0u, 0u, 0u, payload0
    );

    params.image[iy * params.width + ix] = payload0;
}

extern "C" __global__ void __miss__cube() { optixSetPayload_0(0xFF201020u); }

extern "C" __global__ void __closesthit__cube() { optixSetPayload_0(0xFF20A0F0u); }
// -- section: trace_mode_trace0 -- end

// -- section: renderer_launch_params -- begin
struct RendererLaunchParams {
    unsigned int* image;
    unsigned int width;
    unsigned int height;
    unsigned int frame_id;
    unsigned long long trav_handle;
    float cam_px;
    float cam_py;
    float cam_pz;
    float cam_ux;
    float cam_uy;
    float cam_uz;
    float cam_vx;
    float cam_vy;
    float cam_vz;
    float cam_wx;
    float cam_wy;
    float cam_wz;
    unsigned long long instance_vertex_ptrs;
    unsigned long long instance_index_ptrs;
    unsigned long long instance_normal_ptrs;
    unsigned long long instance_color_ptrs;
    float light_dx;
    float light_dy;
    float light_dz;
};

extern "C" {
__constant__ RendererLaunchParams renderer_params;
}
// -- section: renderer_launch_params -- end

// -- section: renderer_trace_programs -- begin
static __forceinline__ __device__ unsigned int _pack_rgba(float r, float g, float b)
{
    r = fminf(fmaxf(r, 0.0f), 1.0f);
    g = fminf(fmaxf(g, 0.0f), 1.0f);
    b = fminf(fmaxf(b, 0.0f), 1.0f);
    const unsigned int ri = (unsigned int)(255.0f * r);
    const unsigned int gi = (unsigned int)(255.0f * g);
    const unsigned int bi = (unsigned int)(255.0f * b);
    return (255u << 24) | (bi << 16) | (gi << 8) | ri;
}

static __forceinline__ __device__ float _dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

static __forceinline__ __device__ float3 _normalize3(float3 v)
{
    const float inv_len = rsqrtf(fmaxf(_dot3(v, v), 1.0e-20f));
    return make_float3(v.x * inv_len, v.y * inv_len, v.z * inv_len);
}

static __forceinline__ __device__ float3 _cross3(float3 a, float3 b)
{
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

static __forceinline__ __device__ float3 _sky_color(float3 d)
{
    const float t = 0.5f * (d.y + 1.0f);
    const float3 horizon = make_float3(0.86f, 0.83f, 0.72f);
    const float3 zenith = make_float3(0.28f, 0.50f, 0.78f);
    float3 sky = make_float3(
        horizon.x + (zenith.x - horizon.x) * t, horizon.y + (zenith.y - horizon.y) * t,
        horizon.z + (zenith.z - horizon.z) * t
    );
    const float3 sun_dir = _normalize3(make_float3(0.4f, 0.8f, 0.2f));
    const float sun = powf(fmaxf(_dot3(d, sun_dir), 0.0f), 256.0f);
    sky = make_float3(sky.x + 1.2f * sun, sky.y + 1.1f * sun, sky.z + 0.9f * sun);
    return sky;
}

extern "C" __global__ void __raygen__renderer()
{
    const uint3 idx = optixGetLaunchIndex();
    const unsigned int ix = idx.x;
    const unsigned int iy = idx.y;
    if (ix >= renderer_params.width || iy >= renderer_params.height)
        return;

    const float fx = (2.0f * ((float)ix + 0.5f) / (float)renderer_params.width - 1.0f);
    const float fy = -(2.0f * ((float)iy + 0.5f) / (float)renderer_params.height - 1.0f);

    float dx = renderer_params.cam_wx + fx * renderer_params.cam_ux + fy * renderer_params.cam_vx;
    float dy = renderer_params.cam_wy + fx * renderer_params.cam_uy + fy * renderer_params.cam_vy;
    float dz = renderer_params.cam_wz + fx * renderer_params.cam_uz + fy * renderer_params.cam_vz;
    const float inv_len = rsqrtf(dx * dx + dy * dy + dz * dz);
    dx *= inv_len;
    dy *= inv_len;
    dz *= inv_len;

    unsigned int payload0 = 0xFF101820u;
    optixTrace(
        renderer_params.trav_handle,
        make_float3(renderer_params.cam_px, renderer_params.cam_py, renderer_params.cam_pz), make_float3(dx, dy, dz),
        0.001f, 1.0e16f, 0.0f, 255u, 0u, 0u, 1u, 0u, payload0
    );

    const unsigned int py = renderer_params.height - 1u - iy;
    renderer_params.image[py * renderer_params.width + ix] = payload0;
}

extern "C" __global__ void __miss__renderer()
{
    const float3 d = _normalize3(optixGetWorldRayDirection());
    const float3 sky = _sky_color(d);
    optixSetPayload_0(_pack_rgba(sky.x, sky.y, sky.z));
}

extern "C" __global__ void __miss__shadow()
{
    // If a shadow ray misses, light is visible.
    optixSetPayload_0(1u);
}

extern "C" __global__ void __closesthit__renderer()
{
    // Fetch mesh data and compute geometric normal.
    const unsigned int iid = optixGetInstanceId();
    const unsigned long long* v64 = (const unsigned long long*)renderer_params.instance_vertex_ptrs;
    const unsigned long long* i64 = (const unsigned long long*)renderer_params.instance_index_ptrs;
    const unsigned long long* n64 = (const unsigned long long*)renderer_params.instance_normal_ptrs;
    const unsigned int* color_table = (const unsigned int*)renderer_params.instance_color_ptrs;
    const float3* vertices = (const float3*)v64[iid];
    const uint3* indices = (const uint3*)i64[iid];
    const float3* normals = (const float3*)n64[iid];
    const unsigned int tri_id = optixGetPrimitiveIndex();
    const uint3 tri = indices[tri_id];

    const float2 bary = optixGetTriangleBarycentrics();
    const float bw = 1.0f - bary.x - bary.y;
    float3 n_obj;
    if (normals != 0) {
        const float3 n0 = normals[tri.x];
        const float3 n1 = normals[tri.y];
        const float3 n2 = normals[tri.z];
        n_obj = make_float3(
            bw * n0.x + bary.x * n1.x + bary.y * n2.x, bw * n0.y + bary.x * n1.y + bary.y * n2.y,
            bw * n0.z + bary.x * n1.z + bary.y * n2.z
        );
        n_obj = _normalize3(n_obj);
    } else {
        const float3 p0 = vertices[tri.x];
        const float3 p1 = vertices[tri.y];
        const float3 p2 = vertices[tri.z];
        const float3 e1 = make_float3(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z);
        const float3 e2 = make_float3(p2.x - p0.x, p2.y - p0.y, p2.z - p0.z);
        n_obj = _normalize3(_cross3(e1, e2));
    }
    float3 n = _normalize3(optixTransformNormalFromObjectToWorldSpace(n_obj));
    const float3 view_dir = _normalize3(
        make_float3(-optixGetWorldRayDirection().x, -optixGetWorldRayDirection().y, -optixGetWorldRayDirection().z)
    );
    if (_dot3(n, view_dir) < 0.0f) {
        n = make_float3(-n.x, -n.y, -n.z);
    }

    const float3 l
        = _normalize3(make_float3(renderer_params.light_dx, renderer_params.light_dy, renderer_params.light_dz));
    float ndotl = fmaxf(_dot3(n, l), 0.0f);

    // Hard shadow ray.
    const float t = optixGetRayTmax();
    const float3 wo = optixGetWorldRayOrigin();
    const float3 wd = optixGetWorldRayDirection();
    const float3 hit_pos = make_float3(wo.x + t * wd.x, wo.y + t * wd.y, wo.z + t * wd.z);
    const float3 shadow_origin = make_float3(hit_pos.x + 0.01f * n.x, hit_pos.y + 0.01f * n.y, hit_pos.z + 0.01f * n.z);
    const float3 tangent
        = _normalize3(_cross3(fabsf(l.z) < 0.999f ? make_float3(0.0f, 0.0f, 1.0f) : make_float3(0.0f, 1.0f, 0.0f), l));
    const float3 bitangent = _cross3(l, tangent);
    const float2 taps[4] = {
        make_float2(-0.55f, -0.25f),
        make_float2(0.55f, -0.35f),
        make_float2(-0.20f, 0.60f),
        make_float2(0.35f, 0.45f),
    };
    const float area_radius = 0.08f;
    float shadow_vis = 0.0f;
    for (int si = 0; si < 4; ++si) {
        const float2 o = taps[si];
        const float3 ls = _normalize3(make_float3(
            l.x + area_radius * (o.x * tangent.x + o.y * bitangent.x),
            l.y + area_radius * (o.x * tangent.y + o.y * bitangent.y),
            l.z + area_radius * (o.x * tangent.z + o.y * bitangent.z)
        ));
        unsigned int light_visible = 0u;
        optixTrace(
            renderer_params.trav_handle, shadow_origin, ls, 0.001f, 1.0e16f, 0.0f, 255u,
            OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
            0u, 1u, 1u, light_visible
        );
        shadow_vis += (light_visible != 0u) ? 1.0f : 0.0f;
    }
    const float shadow = 0.25f * shadow_vis;

    const unsigned int packed_base = color_table[iid];
    const float3 base = make_float3(
        (float)(packed_base & 255u) * (1.0f / 255.0f), (float)((packed_base >> 8) & 255u) * (1.0f / 255.0f),
        (float)((packed_base >> 16) & 255u) * (1.0f / 255.0f)
    );
    // Sky hemisphere fill similar to C# environment contribution.
    const float3 sky_n = _sky_color(n);
    const float hemi = 0.5f * (n.y + 1.0f);
    float3 c = make_float3(
        base.x * ((0.35f + 0.65f * hemi) * sky_n.x + 0.90f * shadow * ndotl),
        base.y * ((0.35f + 0.65f * hemi) * sky_n.y + 0.90f * shadow * ndotl),
        base.z * ((0.35f + 0.65f * hemi) * sky_n.z + 0.90f * shadow * ndotl)
    );

    // Small view-dependent highlight.
    const float3 h = _normalize3(make_float3(l.x + view_dir.x, l.y + view_dir.y, l.z + view_dir.z));
    const float spec = powf(fmaxf(_dot3(n, h), 0.0f), 32.0f) * 0.18f * shadow;
    c = make_float3(c.x + spec, c.y + spec, c.z + spec);

    // Basic tonemap + gamma to match display appearance better.
    c = make_float3(c.x / (1.0f + c.x), c.y / (1.0f + c.y), c.z / (1.0f + c.z));
    c = make_float3(powf(c.x, 1.0f / 2.2f), powf(c.y, 1.0f / 2.2f), powf(c.z, 1.0f / 2.2f));
    optixSetPayload_0(_pack_rgba(c.x, c.y, c.z));
}
// -- section: renderer_trace_programs -- end
