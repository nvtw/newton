**You:**

How to make a full coordinate rigid body simulator as fast as possible in cuda. Be very specific, i'm cuda developper

* * *

**ChatGPT:**

If you're targeting **maximum throughput on NVIDIA GPUs**, you have to stop thinking of the simulator as "physics" and instead think of it as a **streaming sparse linear algebra pipeline**. The fastest rigid body simulators (PhysX GPU, Warp, Flex, Newton GPU prototypes, proprietary robotics engines) are almost entirely organized around maximizing occupancy, memory bandwidth, and minimizing synchronization—not minimizing FLOPs.

For a **full-coordinate rigid body simulator** (12 or 13 DOF/body, quaternion orientation, constraint stabilization), here's the architecture I would use.

* * *

# 1. Choose an algorithm that maps to GPUs

Avoid recursive algorithms:

* Featherstone ABA ❌
* Articulated Body Algorithm ❌
* Recursive LCP solvers ❌

Instead use

* XPBD
* Jacobi constraint projection
* Matrix-free Projected Gauss-Seidel (block Jacobi)
* Conjugate Gradient
* Nonlinear Jacobi

The reason is simple:

GPUs love

```
many independent constraints
```

not

```
dependency chains
```

A million constraints solved with Jacobi is usually much faster than a sophisticated serial algorithm.

* * *

# 2. Data layout

Everything SoA.

Never

```C++
struct RigidBody
{
    float3 pos;
    quat rot;
    ...
};
```

Instead

```C++
float4* position;
float4* rotation;

float4* linearVelocity;
float4* angularVelocity;

float4* force;
float4* torque;

float4* invInertiaWorld0;
float4* invInertiaWorld1;
float4* invInertiaWorld2;

float* invMass;
```

One array per quantity.

This alone often gives 2× bandwidth improvement.

* * *

# 3. 128-byte alignment

Design every array so that

```
thread i

loads exactly

16 bytes
```

Example

```
float4 position

thread0 -> body0
thread1 -> body1
...

warp

loads

512 bytes

fully coalesced
```

Never store float3.

Always

```
float4
```

* * *

# 4. Minimize kernel launches

Bad

```
integrate

launch

contacts

launch

joint

launch

velocity

launch

position

launch
```

Better

```
persistent kernel

while(simulation)
{
    integrate

    broadphase

    narrowphase

    solve

    integrate
}
```

Kernel launches become negligible.

* * *

# 5. Persistent threads

Instead of

```
N threads
```

Launch

```
#SM × resident blocks
```

Then work queue.

Example

```
while(pop(body))
{
    integrate(body);
}
```

Occupancy increases dramatically.

* * *

# 6. Contact generation

Broadphase

Morton code

↓

radix sort

↓

cell offsets

↓

neighbor search

Exactly the SPH pipeline.

Never BVH traversal per thread if bodies move every frame.

Uniform grids dominate until scenes become very sparse.

* * *

# 7. Narrow phase

Store shapes separately.

```
sphere

capsule

box

mesh
```

Dispatch

```
SphereSphere

SphereCapsule

CapsuleCapsule

BoxBox
```

Don't virtual dispatch.

Don't function pointers.

Use compile-time specialization.

* * *

# 8. World inertia

Avoid

```
R * I * Rᵀ
```

every constraint.

Instead compute once per frame.

Store

```
Iworld

IworldInverse
```

Then constraints only perform

```
I^-1 * torque
```

* * *

# 9. Quaternion update

Instead of

```
quat *= exp(ωdt)
```

Use

```
dq = 0.5 * Ω(q) * ω

q += dq * dt

normalize(q)
```

One normalize per body.

* * *

# 10. Constraint batching

Huge speedup.

Color graph.

Constraints sharing bodies

↓

different colors

Each color

```
fully parallel
```

No atomics.

Typical

```
10–30 colors
```

for contact graphs.

* * *

# 11. If coloring is impossible

Use

```
Jacobi

instead of

Gauss-Seidel
```

Each constraint

```
reads

writes delta
```

Then reduce.

Never

```
atomicAdd(position)
```

unless unavoidable.

* * *

# 12. Delta accumulation

Instead of

```
atomicAdd(body.position)
```

Have

```
deltaPosition

deltaRotation
```

Then

```
body += delta
```

One reduction kernel.

Massive reduction in contention.

* * *

# 13. Register pressure

This is usually the #1 bottleneck.

A contact solver easily reaches

```
120 registers
```

Occupancy collapses.

Target

```
48–64 registers/thread
```

Use

```
__launch_bounds__
```

or

```
-maxrregcount
```

Profile both.

* * *

# 14. Warp-level primitives

Never reduce with shared memory.

Instead

```C++
x += __shfl_down_sync(mask,x,16);
```

Then

```
8

4

2

1
```

Shared memory only if necessary.

* * *

# 15. Shared memory

Good for

```
constraint batch

joint cache

small Jacobian
```

Bad for

large body arrays.

* * *

# 16. Solver

The fastest GPU solvers are usually

```
for iteration

    constraint kernel

    reduction kernel
```

or

```
constraint kernel

warp reduction

position update
```

rather than monolithic PGS.

* * *

# 17. Matrix-free Jacobians

Never build

```
J

M

K
```

Explicitly.

Store only

```
bodyA

bodyB

contact normal

lever arms
```

Compute

```
JM^-1Jᵀ
```

on the fly.

Memory traffic dominates FLOPs.

* * *

# 18. Contact cache

Warm starting.

Store

```
impulse

friction impulse

normal impulse
```

Reuse next frame.

Nearly halves iterations.

* * *

# 19. Friction

Solve

```
normal

↓

tangent1

↓

tangent2
```

independently.

GPU likes independent scalar constraints.

* * *

# 20. Mixed precision

Very effective.

Store

```
positions float32

velocity float32

quaternion float32
```

Accumulate

```
dot products

constraint residual

CG reduction
```

in FP64 if long chains or robotics accuracy is required, or use compensated summation within a warp. Otherwise stay entirely in FP32 for games.

* * *

# 21. Occupancy targets

Aim for:

| Metric | Target |
| --- | --- |
| SM occupancy | 60–90% |
| Global load efficiency | >95% |
| L2 hit rate | >70% |
| Warp execution efficiency | >90% |
| Branch efficiency | >95% |
| Registers/thread | <64 |
| DRAM throughput | Near hardware limit |

Use Nsight Compute to drive every optimization; don't optimize by intuition.

* * *

# 22. Recommended simulation pipeline

```
Persistent kernel

    integrate forces

    build Morton codes

    radix sort

    cell offsets

    broadphase

    narrowphase

    build constraint list

    graph coloring (or Jacobi staging)

    warm start

    for iterations

        solve normal

        solve friction

        reduce body deltas

    integrate positions

    update inertia tensors

    sleep detection
```

No CPU synchronization until the frame is complete.

* * *

# 23. Advanced optimizations

Once the basics are saturated, the next level includes:

* **CUDA Graphs** to eliminate launch overhead if you keep a multi-kernel pipeline.
* **`cp.async`** (Ampere+) to stage constraint batches into shared memory while computing the previous batch.
* **Tensor Memory Accelerator (TMA)** on Hopper for large tiled transfers if constraint data is suitably structured.
* **Cluster launches** (SM90+) for large shared working sets spanning thread-block clusters.
* **Cooperative Groups** for persistent global work queues and synchronization.
* **Body reordering** every frame (or every few frames) using Morton order so neighboring constraints access neighboring body indices, improving L2 locality.
* **Compressed contact manifolds** (e.g., 16-byte or 32-byte packed contacts) to maximize cache efficiency.

## If the goal is absolute maximum performance

For simulations with **millions of bodies**, the dominant design is typically:

* Position-based or impulse-based **Jacobi** solver (fully parallel)
* Morton-code uniform-grid broadphase with radix sort
* SoA layout using `float4`-aligned data
* Persistent kernels with global work queues
* Graph-colored constraint batches where practical
* Matrix-free Jacobian evaluation
* Warm-started contacts
* One-time-per-frame world inertia computation
* Body reordering to maximize cache locality
* Warp-level reductions instead of shared-memory reductions
* Aggressive control of register pressure and occupancy

That architecture maps extremely well to modern Ada, Hopper, and Blackwell GPUs because it maximizes memory throughput, minimizes synchronization, and exposes enormous fine-grained parallelism.