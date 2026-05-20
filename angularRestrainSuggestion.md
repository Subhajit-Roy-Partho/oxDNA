Yes. An angular COM restraint is implementable in oxDNA in a way that matches the existing CPU/GPU external-force architecture. The main caveat is terminology: the existing `type = com` is not a holonomic “constraint” in the SHAKE/RATTLE sense; it is a harmonic external spring on one COM relative to another COM. I would implement the angular version the same way: a conservative external restraint potential that applies forces to particles, not an exact constraint enforced by the integrator.

## 1. What oxDNA’s existing COM force actually does

oxDNA’s public repo describes the code as supporting simulations on single CPU cores and NVIDIA GPUs, and the external-force docs say that external forces are enabled through `external_forces = true` plus an external-force file. The docs also warn that external forces can inject work and require care with timestep and thermostat choices. ([github.com](https://github.com/lorenzo-rovigatti/oxDNA))

The documentation for the existing COM force lists exactly four core inputs: `stiff`, `r0`, `com_list`, and `ref_list`. `com_list` is the group whose center of mass is subject to the force, and `ref_list` is the group whose COM is used as the reference. ([lorenzo-rovigatti.github.io](https://lorenzo-rovigatti.github.io/oxDNA/forces.html))

### CPU path

The CPU force is `COMForce`, derived from `BaseForce`. Its header describes it as “a force acting on the centre of mass of an ensemble of particles,” and it stores `_com_list`, `_ref_list`, `_r0`, `_last_step`, `_com`, and `_ref_com`. ([github.com](https://github.com/lorenzo-rovigatti/oxDNA/blob/master/src/Forces/COMForce.h))

The force is wired into parsing through `ForceFactory`: `type = com` becomes `std::make_shared<COMForce>()`; then `init()` returns a particle-index list and `CONFIG_INFO->add_force_to_particles()` attaches that force to those particles. The same factory also already knows about a metadynamics-specific `meta_com_angle_trap`, which matters later. ([github.com](https://github.com/lorenzo-rovigatti/oxDNA/blob/master/src/Forces/ForceFactory.cpp))

`COMForce::init()` reads `com_list`, `ref_list`, `stiff`, `r0`, and optional `rate`. It parses the two particle lists, stores the corresponding `BaseParticle*` pointers, and returns only `com_indexes`, not `ref_indexes`. This means the force is attached only to particles in `com_list`; the reference COM is not pulled back unless the user defines another force. ([github.com](https://github.com/lorenzo-rovigatti/oxDNA/blob/master/src/Forces/COMForce.cpp)) That non-reciprocal pattern is consistent with the docs’ warning for `mutual_trap`: the “reference” particle does not automatically feel the opposite force. ([lorenzo-rovigatti.github.io](https://lorenzo-rovigatti.github.io/oxDNA/forces.html))

At runtime, `ConfigInfo::add_force_to_particles()` stores the shared force object and adds a raw pointer to each target particle’s `ext_forces` vector. Then each `BaseParticle` computes its external contribution by looping over `ext_forces` and adding `ext_force->value(step, abs_pos)` to its force; potential energy is accumulated analogously through `potential()`. ([github.com](https://github.com/lorenzo-rovigatti/oxDNA/blob/master/src/Utilities/ConfigInfo.cpp))

`COMForce::_compute_coms()` caches the two COMs by timestep. It computes arithmetic means, not mass-weighted means, using `CONFIG_INFO->box->get_abs_pos(p)` for every particle in each group. The metadynamics helper `particle_list_com()` uses the same absolute-position averaging pattern. ([github.com](https://github.com/lorenzo-rovigatti/oxDNA/blob/master/src/Forces/COMForce.cpp))

Mathematically, the CPU force is:

\[
\mathbf{R}_C = \frac{1}{N_C}\sum_{i\in C}\mathbf{r}_i,\qquad
\mathbf{R}_R = \frac{1}{N_R}\sum_{j\in R}\mathbf{r}_j
\]

\[
\mathbf{d} = \mathbf{R}_R-\mathbf{R}_C,\qquad d=\|\mathbf{d}\|,\qquad
r_\star(t)=r_0+\mathrm{rate}\cdot t
\]

\[
E_\mathrm{total}=\frac{1}{2}k(d-r_\star)^2
\]

For each particle in `com_list`, the code returns:

\[
\mathbf{F}_i
=
\frac{k}{N_C}(d-r_\star)\frac{\mathbf{d}}{d}
\]

and `potential()` returns:

\[
E_i=\frac{1}{N_C}E_\mathrm{total}
\]

so that summing the per-particle external potential over all particles in `com_list` gives the total spring energy. The source does this directly: `value()` divides by `_com_list.size()`, and `potential()` divides the harmonic energy by `_com_list.size()`. ([github.com](https://github.com/lorenzo-rovigatti/oxDNA/blob/master/src/Forces/COMForce.cpp))

One important numerical issue: `COMForce::value()` divides by `d_com` without guarding `d_com == 0`. If the COM and reference COM coincide exactly, that path can produce a division by zero. ([github.com](https://github.com/lorenzo-rovigatti/oxDNA/blob/master/src/Forces/COMForce.cpp)) Do not copy that weakness into the angular implementation.

### GPU path

On CUDA, external forces are copied into a `CUDA_trap` union. The current CUDA force enum defines `CUDA_COM_FORCE` as 13, and the `COM_force` struct stores `stiff`, `r0`, `rate`, `n_com`, `n_ref`, and device pointers for `com_indexes` and `ref_indexes`. ([github.com](https://github.com/lorenzo-rovigatti/oxDNA/blob/master/src/CUDA/CUDAForces.h))

The host-side conversion function `init_COMForce_from_CPU()` copies the CPU-side `COMForce` parameters and particle indexes into the CUDA struct. It allocates device arrays for the COM and reference indexes on first initialization, then copies host vectors to those arrays. The current code even contains a `TODO` noting that this memory is not freed. ([github.com](https://github.com/lorenzo-rovigatti/oxDNA/blob/master/src/CUDA/CUDAForces.h))

`MD_CUDABackend::_apply_external_forces_changes()` rejects external forces when CUDA sorting is enabled, builds a host vector of `CUDA_trap` objects indexed by force slot and particle, and dispatches `COMForce` through `init_COMForce_from_CPU()`. The same code path throws an error if a force type is not in the CUDA-supported list, so a new angular force must be added there too. ([github.com](https://github.com/lorenzo-rovigatti/oxDNA/blob/master/src/CUDA/Backends/MD_CUDABackend.cu))

In the CUDA kernel path, `case CUDA_COM_FORCE` recomputes the two COMs by looping over the stored device indexes, averaging `poss[p_idx]`, forming `dr = ref - com`, computing `dr_abs`, and adding the same spring force divided by `n_com` to the current particle’s force. ([github.com](https://github.com/lorenzo-rovigatti/oxDNA/blob/master/src/CUDA/Backends/CUDA_MD.cuh))

The GPU version therefore mirrors the CPU formula, but it computes the COMs independently inside every particle/thread that has that force. That is simple but redundant. For an angular COM force, you can follow this pattern first, then optimize later.

---

## 2. Is an angular COM restraint meaningful in oxDNA?

Yes, with a precise definition. The meaningful quantity is the unsigned bend angle formed by three group centers:

\[
A \;-
B \;-
C
\]

where `B` is the vertex group. This is not a nucleotide-body orientation torque and not a twist around a DNA axis. It is a translational force field derived from a potential of the three COM positions.

That distinction matters because oxDNA’s external-force docs state that forces act on nucleotide centers of mass and that generic torques or site-specific forces are not supported through the standard external-force interface. ([lorenzo-rovigatti.github.io](https://lorenzo-rovigatti.github.io/oxDNA/forces.html)) The CPU particle loop also calls only `BaseForce::value()` and adds the returned vector to `force`; it does not ask external forces for a torque. ([github.com](https://github.com/lorenzo-rovigatti/oxDNA/blob/master/src/Particles/BaseParticle.cpp))

There is already repo precedent for a COM-angle-like object: `ForceFactory` includes and registers `LTCOMAngleTrap` as `type = meta_com_angle_trap`. ([github.com](https://github.com/lorenzo-rovigatti/oxDNA/blob/master/src/Forces/ForceFactory.cpp)) I would treat `LTCOMAngleTrap` as proof that the concept fits oxDNA, but I would not copy its math directly for a general-purpose angular restraint. It uses `acos(dot_product)` and a denominator involving the dot product in the force prefactor, so it has exactly the near-collinear numerical sensitivity you should avoid in a stable new force. ([github.com](https://github.com/lorenzo-rovigatti/oxDNA/blob/master/src/Forces/Metadynamics/LTCOMAngleTrap.cpp))

The broader MD ecosystem supports the same physical idea. PLUMED defines a three-point angle as

\[
\theta=\arccos\left(\frac{\mathbf{r}_{21}\cdot\mathbf{r}_{23}}
{|\mathbf{r}_{21}|\,|\mathbf{r}_{23}|}\right)
\]

and notes that such angles live in `[0,\pi]`, not on a periodic circle. ([plumed.org](https://www.plumed.org/doc-v2.9/user-doc/html/_a_n_g_l_e.html)) GROMACS documents both harmonic angle potentials in `\theta` and cosine-based angle potentials in `\cos\theta-\cos\theta_0`. ([manual.gromacs.org](https://manual.gromacs.org/current/reference-manual/functions/bonded-interactions.html))

For oxDNA, I recommend a cosine-based COM-angle restraint as the default because it avoids the `1/\sin\theta` singularity that appears when differentiating `acos`.

---

## 3. Recommended mathematical formulation

Define three non-overlapping particle groups:

- `A`: first arm group.
- `B`: vertex group.
- `C`: second arm group.

Use arithmetic COMs to match `COMForce`:

\[
\mathbf{R}_A = \frac{1}{N_A}\sum_{i\in A}\mathbf{r}_i,\quad
\mathbf{R}_B = \frac{1}{N_B}\sum_{i\in B}\mathbf{r}_i,\quad
\mathbf{R}_C = \frac{1}{N_C}\sum_{i\in C}\mathbf{r}_i
\]

Then define the two arms:

\[
\mathbf{u}=\mathbf{R}_A-\mathbf{R}_B,\qquad
\mathbf{v}=\mathbf{R}_C-\mathbf{R}_B
\]

\[
r_u=\|\mathbf{u}\|,\qquad r_v=\|\mathbf{v}\|
\]

\[
\hat{\mathbf{u}}=\frac{\mathbf{u}}{r_u},\qquad
\hat{\mathbf{v}}=\frac{\mathbf{v}}{r_v}
\]

\[
c=\hat{\mathbf{u}}\cdot\hat{\mathbf{v}}=\cos\theta
\]

Clamp only for roundoff:

\[
c \leftarrow \min(1,\max(-1,c))
\]

Let the target angle be:

\[
\theta_\star(t)=\theta_0+\mathrm{rate}\cdot t
\]

and:

\[
c_\star(t)=\cos\theta_\star(t)
\]

### Default stable potential

Use:

\[
E=\frac{1}{2}k_c\left(c-c_\star\right)^2
\]

This is the cosine-based angle potential. It is stable because the force requires division by `r_u` and `r_v`, but not by `\sin\theta`. The only truly undefined case is when either arm length goes to zero.

If you want the input `stiff` to behave like a normal angular stiffness `k_\theta` near the target, use the small-angle relation:

\[
c-c_\star \approx -\sin(\theta_\star)(\theta-\theta_\star)
\]

so near the target:

\[
E\approx \frac{1}{2}k_c\sin^2(\theta_\star)(\theta-\theta_\star)^2
\]

Therefore choose:

\[
k_c=\frac{k_\theta}{\max(\sin^2\theta_\star,\sin^2\theta_\mathrm{min})}
\]

where `\theta_\mathrm{min}` is a small cutoff such as `5^\circ` in radians. This gives intuitive radian-stiffness behavior for ordinary bend angles while preventing `k_c` from exploding near `0` or `\pi`.

For targets very close to `0` or `\pi`, I would not use the stiffness conversion above. Instead use endpoint-specific potentials:

For alignment, `\theta_\star=0`:

\[
E=k(1-c)
\]

For anti-alignment, `\theta_\star=\pi`:

\[
E=k(1+c)
\]

These have clean gradients and do not pretend that a quadratic angular well is well-defined at an exactly collinear point.

### Force derivation for the default cosine potential

Let:

\[
q=c-c_\star
\]

The derivatives of `c` with respect to the two arm vectors are:

\[
\frac{\partial c}{\partial \mathbf{u}}
=
\frac{\hat{\mathbf{v}}-c\hat{\mathbf{u}}}{r_u}
\]

\[
\frac{\partial c}{\partial \mathbf{v}}
=
\frac{\hat{\mathbf{u}}-c\hat{\mathbf{v}}}{r_v}
\]

Since `\mathbf{u}=\mathbf{R}_A-\mathbf{R}_B` and `\mathbf{v}=\mathbf{R}_C-\mathbf{R}_B`:

\[
\frac{\partial c}{\partial \mathbf{R}_A}
=
\frac{\hat{\mathbf{v}}-c\hat{\mathbf{u}}}{r_u}
\]

\[
\frac{\partial c}{\partial \mathbf{R}_C}
=
\frac{\hat{\mathbf{u}}-c\hat{\mathbf{v}}}{r_v}
\]

\[
\frac{\partial c}{\partial \mathbf{R}_B}
=
-
\frac{\partial c}{\partial \mathbf{R}_A}
-
\frac{\partial c}{\partial \mathbf{R}_C}
\]

For

\[
E=\frac{1}{2}k_c q^2
\]

the COM-level forces are:

\[
\mathbf{F}_A
=
-k_c q
\frac{\hat{\mathbf{v}}-c\hat{\mathbf{u}}}{r_u}
\]

\[
\mathbf{F}_C
=
-k_c q
\frac{\hat{\mathbf{u}}-c\hat{\mathbf{v}}}{r_v}
\]

\[
\mathbf{F}_B
=-(\mathbf{F}_A+\mathbf{F}_C)
\]

That last identity is important: the angular restraint has exactly zero net force on the three-COM system. It can bend the configuration, but it does not translate the whole system.

Then distribute group forces uniformly over particles:

\[
\mathbf{f}_i=\frac{\mathbf{F}_A}{N_A}\quad i\in A
\]

\[
\mathbf{f}_i=\frac{\mathbf{F}_B}{N_B}\quad i\in B
\]

\[
\mathbf{f}_i=\frac{\mathbf{F}_C}{N_C}\quad i\in C
\]

This mirrors the existing COM force’s “compute a COM-level force, divide by group size” pattern.

### Optional exact harmonic-in-angle mode

You can also implement:

\[
E=\frac{1}{2}k_\theta(\theta-\theta_\star)^2
\]

with:

\[
\theta=\arccos(c)
\]

or, more stably for evaluating the angle,

\[
\theta=\operatorname{atan2}\left(\|\hat{\mathbf{u}}\times\hat{\mathbf{v}}\|,c\right)
\]

The force requires:

\[
\sin\theta=\sqrt{1-c^2}
\]

and:

\[
\mathbf{F}_A
=
 k_\theta(\theta-\theta_\star)
\frac{\hat{\mathbf{v}}-c\hat{\mathbf{u}}}{r_u\sin\theta}
\]

\[
\mathbf{F}_C
=
 k_\theta(\theta-\theta_\star)
\frac{\hat{\mathbf{u}}-c\hat{\mathbf{v}}}{r_v\sin\theta}
\]

\[
\mathbf{F}_B=-(\mathbf{F}_A+\mathbf{F}_C)
\]

This is physically familiar, but it is not my recommended default because the force direction is singular when `\theta\to 0` or `\theta\to\pi`. Use it only with a strict `sin_min` guard, and reject or warn for near-collinear configurations.

---

## 4. Input syntax I would add

Use names that make the vertex unambiguous:

```text
{
    type = com_angle

    # Angle is group1 - vertex_group - group2
    group1 = 0-9
    vertex_group = 10-19
    group2 = 20-29

    # radians, not degrees
    theta0 = 1.5707963267948966

    # interpreted according to mode; see below
    stiff = 1.0

    # optional, radians per MD/MC step
    rate = 0.0

    # recommended default
    mode = cosine

    # optional: if true, use box->min_image between COMs
    PBC = false

    # minimum allowed COM-arm length
    min_arm = 1e-6
}
```

I would support aliases later, but start with one clear syntax. Avoid `p1a/p2a/p3a` unless you intentionally want to mimic the metadynamics naming.

Document the angle as unsigned and constrained to `[0,\pi]`. PLUMED makes the same point for this standard angle definition. ([plumed.org](https://www.plumed.org/doc-v2.9/user-doc/html/_a_n_g_l_e.html))

---

## 5. CPU implementation plan, step by step

### Step 1: Add new source files

Create:

```text
src/Forces/COMAngleForce.h
src/Forces/COMAngleForce.cpp
```

The build system uses an explicit `forces_SOURCES` list in `src/CMakeLists.txt`, which already lists `Forces/COMForce.cpp` and `Forces/Metadynamics/LTCOMAngleTrap.cpp`; add `Forces/COMAngleForce.cpp` there. ([github.com](https://github.com/lorenzo-rovigatti/oxDNA/blob/master/src/CMakeLists.txt))

### Step 2: Define the class

Derive from `BaseForce`, just like `COMForce`.

Members I would include:

```cpp
class COMAngleForce : public BaseForce {
protected:
    llint _last_step = -1;

    std::string _group1_string;
    std::string _vertex_group_string;
    std::string _group2_string;

    std::vector<int> _group1_indexes;
    std::vector<int> _vertex_indexes;
    std::vector<int> _group2_indexes;

    std::set<BaseParticle *> _group1;
    std::set<BaseParticle *> _vertex_group;
    std::set<BaseParticle *> _group2;

    // Fast role lookup by particle index.
    // 0 = none, 1 = group1, 2 = vertex, 3 = group2.
    std::unordered_map<int, int> _role_by_index;

    number _theta0 = 0.0;
    number _min_arm = 1e-6;
    number _sin2_min = 1e-4;
    bool _PBC = false;

    enum Mode {
        COSINE = 0,
        THETA = 1,
        ALIGN = 2,
        ANTI_ALIGN = 3
    };

    int _mode = COSINE;

    LR_vector _RA, _RB, _RC;
    LR_vector _FA, _FB, _FC;
    number _energy = 0.0;

    void _compute(llint step);
    int _role_of_current_particle() const;

public:
    COMAngleForce();
    virtual ~COMAngleForce();

    virtual std::tuple<std::vector<int>, std::string> init(input_file &inp) override;
    virtual LR_vector value(llint step, LR_vector &pos) override;
    virtual number potential(llint step, LR_vector &pos) override;
};
```

`BaseForce` already has `_current_particle` and `set_current_particle(BaseParticle*)`, but the current `BaseParticle` loop does not call it before `value()` or `potential()`. ([github.com](https://github.com/lorenzo-rovigatti/oxDNA/blob/master/src/Forces/BaseForce.h)) You need to fix that for a single three-group angular force object to know which group the current particle belongs to.

### Step 3: Modify `BaseParticle` to set the current particle

In `BaseParticle::set_initial_forces()` change the loop to:

```cpp
for(auto ext_force : ext_forces) {
    ext_force->set_current_particle(this);
    force += ext_force->value(step, abs_pos);
}
```

In `BaseParticle::set_ext_potential()` change the loop to:

```cpp
for(auto ext_force : ext_forces) {
    ext_force->set_current_particle(this);
    ext_potential += ext_force->potential(step, abs_pos);
}
```

Existing force classes should ignore `_current_particle`, so this is a low-risk interface activation rather than a new API.

### Step 4: Parse and validate input in `init()`

In `COMAngleForce::init(input_file &inp)`:

1. Call `BaseForce::init(inp)`.
2. Read `group1`, `vertex_group`, `group2`.
3. Read `stiff`.
4. Read `theta0`.
5. Read optional `rate`.
6. Read optional `mode`, defaulting to `cosine`.
7. Read optional `PBC`, defaulting to false.
8. Read optional `min_arm`, defaulting to something like `1e-6`.

Use `Utils::get_particles_from_string()` exactly as `COMForce` does. ([github.com](https://github.com/lorenzo-rovigatti/oxDNA/blob/master/src/Forces/COMForce.cpp))

Validation rules:

```text
group1 must be nonempty
vertex_group must be nonempty
group2 must be nonempty
stiff >= 0
min_arm > 0
theta0 in [0, pi]
no particle may appear in more than one of the three groups
```

Disallowing overlaps is strongly recommended. If a particle belongs to two groups, its coordinate contributes to two COMs, and the simple per-role force distribution is no longer correct.

Return the union of all three particle-index lists, not just one group. Unlike `COMForce`, this angular restraint must apply forces to all three COM groups.

### Step 5: Register the force in `ForceFactory`

Add:

```cpp
#include "COMAngleForce.h"
```

near the `COMForce.h` include.

Then add:

```cpp
else if(type_str.compare("com_angle") == 0) {
    extF = std::make_shared<COMAngleForce>();
}
```

near the `type = com` branch. The factory currently maps string force types to concrete classes and then calls `extF->init(inp)` before attaching the returned particle IDs. ([github.com](https://github.com/lorenzo-rovigatti/oxDNA/blob/master/src/Forces/ForceFactory.cpp))

### Step 6: Implement COM computation

Use the same arithmetic averaging convention as `COMForce`:

```cpp
LR_vector COMAngleForce::_particle_list_com(const std::set<BaseParticle *> &group) {
    LR_vector out(0, 0, 0);
    for(auto p : group) {
        out += CONFIG_INFO->box->get_abs_pos(p);
    }
    return out / (number) group.size();
}
```

For vectors, use:

```cpp
LR_vector u;
LR_vector v;

if(_PBC) {
    u = CONFIG_INFO->box->min_image(_RB, _RA);
    v = CONFIG_INFO->box->min_image(_RB, _RC);
}
else {
    u = _RA - _RB;
    v = _RC - _RB;
}
```

The existing metadynamics angle trap has a `_distance(u, v)` helper that uses `box->min_image(u, v)` when `PBC` is true and `v - u` otherwise. ([github.com](https://github.com/lorenzo-rovigatti/oxDNA/blob/master/src/Forces/Metadynamics/LTCOMAngleTrap.cpp))

Be careful with COMs and PBC. PLUMED’s `CENTER` docs explicitly warn that centers under periodic boundaries require atoms to be in the proper periodic image; otherwise the computed center can be wrong. ([plumed.org](https://www.plumed.org/doc-v2.9/user-doc/html/_c_e_n_t_e_r.html)) In oxDNA, matching `COMForce` means using absolute positions by default.

### Step 7: Implement `_compute(step)`

Pseudo-code:

```cpp
void COMAngleForce::_compute(llint step) {
    if(step == _last_step) return;

    _RA = _com(_group1);
    _RB = _com(_vertex_group);
    _RC = _com(_group2);

    LR_vector u = _PBC ? CONFIG_INFO->box->min_image(_RB, _RA) : (_RA - _RB);
    LR_vector v = _PBC ? CONFIG_INFO->box->min_image(_RB, _RC) : (_RC - _RB);

    number ru = u.module();
    number rv = v.module();

    if(ru < _min_arm || rv < _min_arm) {
        throw oxDNAException(
            "COMAngleForce: undefined angle because one COM arm is shorter than min_arm"
        );
    }

    LR_vector a = u / ru;
    LR_vector b = v / rv;

    number c = a * b;
    c = std::max((number)-1.0, std::min((number)1.0, c));

    number theta_target = _theta0 + _rate * step;

    if(theta_target < 0.0 || theta_target > M_PI) {
        throw oxDNAException("COMAngleForce: target angle outside [0, pi]");
    }

    if(_mode == COSINE) {
        number c0 = std::cos(theta_target);

        number sin2 = std::sin(theta_target);
        sin2 *= sin2;

        // If you want stiff to mean angular stiffness near the target:
        number k_eff = _stiff / std::max(sin2, _sin2_min);

        // If you want stiff to mean cosine stiffness directly:
        // number k_eff = _stiff;

        number q = c - c0;

        _energy = 0.5 * k_eff * q * q;

        LR_vector dc_dRA = (b - a * c) / ru;
        LR_vector dc_dRC = (a - b * c) / rv;

        _FA = dc_dRA * (-k_eff * q);
        _FC = dc_dRC * (-k_eff * q);
        _FB = (_FA + _FC) * (-1.0);
    }

    _last_step = step;
}
```

For endpoint modes:

```cpp
if(_mode == ALIGN) {
    _energy = _stiff * (1.0 - c);

    LR_vector dc_dRA = (b - a * c) / ru;
    LR_vector dc_dRC = (a - b * c) / rv;

    _FA = dc_dRA * _stiff;
    _FC = dc_dRC * _stiff;
    _FB = (_FA + _FC) * (-1.0);
}
```

because \(E=k(1-c)\), so \(\mathbf{F}=+k\nabla c\).

For anti-alignment:

```cpp
if(_mode == ANTI_ALIGN) {
    _energy = _stiff * (1.0 + c);

    LR_vector dc_dRA = (b - a * c) / ru;
    LR_vector dc_dRC = (a - b * c) / rv;

    _FA = dc_dRA * (-_stiff);
    _FC = dc_dRC * (-_stiff);
    _FB = (_FA + _FC) * (-1.0);
}
```

because \(E=k(1+c)\), so \(\mathbf{F}=-k\nabla c\).

### Step 8: Implement `value()`

Use `_current_particle` to return the right group contribution:

```cpp
LR_vector COMAngleForce::value(llint step, LR_vector &pos) {
    _compute(step);

    if(_current_particle == nullptr) {
        return LR_vector(0, 0, 0);
    }

    auto it = _role_by_index.find(_current_particle->index);
    if(it == _role_by_index.end()) {
        return LR_vector(0, 0, 0);
    }

    if(it->second == 1) {
        return _FA / (number) _group1.size();
    }
    else if(it->second == 2) {
        return _FB / (number) _vertex_group.size();
    }
    else if(it->second == 3) {
        return _FC / (number) _group2.size();
    }

    return LR_vector(0, 0, 0);
}
```

Do not use `pos`; `COMForce::value()` also ignores the individual particle position because the force is computed from group COMs. ([github.com](https://github.com/lorenzo-rovigatti/oxDNA/blob/master/src/Forces/COMForce.cpp))

### Step 9: Implement `potential()`

Return the total energy divided by the number of attached particles:

```cpp
number COMAngleForce::potential(llint step, LR_vector &pos) {
    _compute(step);

    number n_total =
        (number)(_group1.size() + _vertex_group.size() + _group2.size());

    return _energy / n_total;
}
```

Because the force is attached to every particle in the three groups, summing `ext_potential` over those particles gives the full angular energy. This is the direct analogue of `COMForce::potential()` dividing by `_com_list.size()`. ([github.com](https://github.com/lorenzo-rovigatti/oxDNA/blob/master/src/Forces/COMForce.cpp))

---

## 6. GPU implementation plan

The GPU implementation should mirror the CPU math exactly, but it has one additional design issue: each particle/thread needs to know whether it belongs to group `A`, `B`, or `C`. The existing COM force does not need that because every attached particle receives the same `com_list` force. The angular force has three different role forces.

### Step 1: Add a CUDA enum

In `src/CUDA/CUDAForces.h`, current CUDA force IDs go through `CUDA_REPULSIVE_KEPLER_POINSOT 18`. ([github.com](https://github.com/lorenzo-rovigatti/oxDNA/blob/master/src/CUDA/CUDAForces.h)) Add:

```cpp
#define CUDA_COM_ANGLE_FORCE 19
```

### Step 2: Add a CUDA struct

Add something like:

```cpp
struct COM_angle_force {
    int type;

    c_number stiff;
    c_number theta0;
    c_number rate;
    c_number min_arm;
    c_number sin2_min;

    int mode;

    // Critical: role of the current particle/trap.
    // 1 = group1, 2 = vertex, 3 = group2.
    int role;

    int n_group1;
    int n_vertex;
    int n_group2;

    int *group1_indexes;
    int *vertex_indexes;
    int *group2_indexes;
};
```

Then add it to `union CUDA_trap`, just like `COM_force comforce;` is already present. ([github.com](https://github.com/lorenzo-rovigatti/oxDNA/blob/master/src/CUDA/CUDAForces.h))

### Step 3: Add a CPU-to-CUDA initializer

Create:

```cpp
void init_COMAngleForce_from_CPU(
    COM_angle_force *cuda_force,
    COMAngleForce *cpu_force,
    BaseParticle *current_particle,
    bool first_time
)
```

It should:

1. Set `cuda_force->type = CUDA_COM_ANGLE_FORCE`.
2. Copy `stiff`, `theta0`, `rate`, `min_arm`, `sin2_min`, and `mode`.
3. Copy group sizes.
4. Set `role` by asking the CPU force which group `current_particle->index` belongs to.
5. Copy group index arrays to device memory.

For a prototype, you can imitate `init_COMForce_from_CPU()`, which builds local index vectors, allocates device arrays on `first_time`, and copies them. ([github.com](https://github.com/lorenzo-rovigatti/oxDNA/blob/master/src/CUDA/CUDAForces.h))

For a production-quality implementation, avoid repeating the current COM memory-management problem. The existing CUDA COM initializer has a `TODO` saying the allocated memory is never freed. ([github.com](https://github.com/lorenzo-rovigatti/oxDNA/blob/master/src/CUDA/CUDAForces.h)) A better design is to cache one set of device arrays per `COMAngleForce*` and reuse those pointers for every particle’s `CUDA_trap`, then free them in the CUDA backend destructor.

### Step 4: Register the CUDA type in `MD_CUDABackend`

In `_apply_external_forces_changes()`, add a branch next to the `COMForce` branch:

```cpp
else if(force_type == typeid(COMAngleForce)) {
    COMAngleForce *p_force = (COMAngleForce *) p->ext_forces[j];
    init_COMAngleForce_from_CPU(
        &cuda_force->comangleforce,
        p_force,
        p,
        first_time
    );
}
```

Also update the error message listing supported CUDA forces. The current message explicitly lists supported force types and includes `COMForce` and `LTCOMTrap`. ([github.com](https://github.com/lorenzo-rovigatti/oxDNA/blob/master/src/CUDA/Backends/MD_CUDABackend.cu))

Do not try to support this with CUDA sorting enabled at first. The backend currently rejects any external force with `_sort_every > 0`. ([github.com](https://github.com/lorenzo-rovigatti/oxDNA/blob/master/src/CUDA/Backends/MD_CUDABackend.cu))

### Step 5: Add the kernel case

In `src/CUDA/Backends/CUDA_MD.cuh`, add:

```cpp
case CUDA_COM_ANGLE_FORCE: {
    ...
    break;
}
```

Use the same structure as `case CUDA_COM_FORCE`, which loops over group indexes, averages positions, computes the force, and adds it to `F`. ([github.com](https://github.com/lorenzo-rovigatti/oxDNA/blob/master/src/CUDA/Backends/CUDA_MD.cuh))

Pseudo-code:

```cpp
case CUDA_COM_ANGLE_FORCE: {
    c_number4 RA = make_c_number4(0., 0., 0., 0.);
    c_number4 RB = make_c_number4(0., 0., 0., 0.);
    c_number4 RC = make_c_number4(0., 0., 0., 0.);

    for(int idx = 0; idx < extF.comangle.n_group1; idx++) {
        RA += poss[extF.comangle.group1_indexes[idx]];
    }
    RA.x /= extF.comangle.n_group1;
    RA.y /= extF.comangle.n_group1;
    RA.z /= extF.comangle.n_group1;

    for(int idx = 0; idx < extF.comangle.n_vertex; idx++) {
        RB += poss[extF.comangle.vertex_indexes[idx]];
    }
    RB.x /= extF.comangle.n_vertex;
    RB.y /= extF.comangle.n_vertex;
    RB.z /= extF.comangle.n_vertex;

    for(int idx = 0; idx < extF.comangle.n_group2; idx++) {
        RC += poss[extF.comangle.group2_indexes[idx]];
    }
    RC.x /= extF.comangle.n_group2;
    RC.y /= extF.comangle.n_group2;
    RC.z /= extF.comangle.n_group2;

    c_number4 u = RA - RB;
    c_number4 v = RC - RB;

    c_number ru = _module(u);
    c_number rv = _module(v);

    if(ru > extF.comangle.min_arm && rv > extF.comangle.min_arm) {
        c_number4 a = u / ru;
        c_number4 b = v / rv;

        c_number c = a.x*b.x + a.y*b.y + a.z*b.z;
        c = fmin((c_number)1.0, fmax((c_number)-1.0, c));

        c_number theta_target =
            extF.comangle.theta0 + extF.comangle.rate * (c_number) step;

        c_number c0 = cos(theta_target);

        c_number s0 = sin(theta_target);
        c_number sin2 = s0 * s0;

        c_number k_eff =
            extF.comangle.stiff / fmax(sin2, extF.comangle.sin2_min);

        c_number q = c - c0;

        c_number4 dc_dRA = (b - a * c) / ru;
        c_number4 dc_dRC = (a - b * c) / rv;

        c_number4 FA = dc_dRA * (-k_eff * q);
        c_number4 FC = dc_dRC * (-k_eff * q);
        c_number4 FB = (FA + FC) * ((c_number)-1.0);

        c_number4 force;

        if(extF.comangle.role == 1) {
            force = FA / extF.comangle.n_group1;
        }
        else if(extF.comangle.role == 2) {
            force = FB / extF.comangle.n_vertex;
        }
        else {
            force = FC / extF.comangle.n_group2;
        }

        F.x += force.x;
        F.y += force.y;
        F.z += force.z;
    }

    break;
}
```

For GPU safety, do not throw inside the kernel. If an arm becomes shorter than `min_arm`, add zero force for that step and rely on CPU-side initialization and tests to ensure that this never happens in intended production runs.

---

## 7. Numerical stability rules you should enforce

1. **Reject zero-length arms.** If `\|\mathbf{R}_A-\mathbf{R}_B\| < \texttt{min_arm}` or `\|\mathbf{R}_C-\mathbf{R}_B\| < \texttt{min_arm}`, the angle is undefined.

2. **Clamp dot products.** Always clamp `c` into `[-1,1]` before computing energies or target comparisons. This prevents roundoff from making `acos`, `sqrt(1-c*c)`, or logic invalid.

3. **Prefer cosine mode.** The default force should not divide by `\sin\theta`. This is the main reason to use `E=\frac{1}{2}k(c-c_0)^2`.

4. **Do not expect exact collinear configurations to bend themselves.** In cosine mode, an exactly collinear configuration can have zero gradient even when the target is non-collinear. That is not a bug; the bending direction is geometrically undefined. Start from a slightly non-collinear configuration or define a signed/reference-plane angle if you need deterministic bending out of a straight line.

5. **Disallow overlapping groups.** The simple force distribution assumes each particle contributes to exactly one COM.

6. **Use radians.** Store `theta0` and `rate` in radians and radians per step. If you add `degrees = true`, convert once in `init()` and store radians internally.

7. **Be explicit about PBC.** Default to the same absolute-coordinate convention as `COMForce`. Add `PBC = true` only after testing, because COMs under PBC can be ambiguous if a group straddles a boundary.

8. **Do not call it a torque.** This is a position-derived external force on group COMs. It can generate bending moments through distributed forces, but it is not a direct rigid-body torque on nucleotide orientation.

---

## 8. Test plan before trusting it

Minimum CPU tests:

1. **Zero-energy test.**  
   One-particle groups:
   \[
   A=(1,0,0),\quad B=(0,0,0),\quad C=(0,1,0)
   \]
   with `\theta_0=\pi/2`. Energy and forces should be zero.

2. **Finite-difference gradient test.**  
   Perturb each Cartesian coordinate by `\epsilon`, recompute total energy, and check:
   \[
   F_x \approx -\frac{E(x+\epsilon)-E(x-\epsilon)}{2\epsilon}
   \]

3. **Net-force conservation.**  
   Sum all particle forces from the restraint. It should be zero to numerical precision:
   \[
   \sum_i \mathbf{f}_i = 0
   \]

4. **Energy accounting test.**  
   Sum `ext_potential` over all attached particles and confirm it equals the total `E` computed inside `_compute()`.

5. **Multi-particle COM test.**  
   Use two or more particles per group, arranged so the COMs are known analytically. Confirm forces are divided equally within each group.

6. **Near-collinear test.**  
   Start near `\theta=10^{-6}` or `\pi-10^{-6}`. Confirm no NaNs in cosine mode.

7. **Short-arm validation test.**  
   Put `A` and `B` at the same COM and confirm the CPU path throws a clear error.

Minimum GPU tests:

1. **CPU/GPU parity.**  
   Same one-particle and multi-particle cases, compare forces after one force evaluation.

2. **Role test.**  
   Verify particles in group 1 receive `\mathbf{F}_A/N_A`, vertex particles receive `\mathbf{F}_B/N_B`, and group 2 particles receive `\mathbf{F}_C/N_C`.

3. **No NaN test.**  
   Near-collinear and short-arm fallback cases should not produce NaNs.

4. **Memory test.**  
   Run with CUDA memory checking after repeated initialization or oxpy-style reinitialization, especially because the existing COM CUDA path already has a memory-freeing TODO. ([github.com](https://github.com/lorenzo-rovigatti/oxDNA/blob/master/src/CUDA/CUDAForces.h))

---

## 9. Suggested implementation order

1. Implement CPU `COMAngleForce` with `mode = cosine` only.
2. Add the `BaseParticle::set_current_particle(this)` calls.
3. Register `type = com_angle` in `ForceFactory`.
4. Add the source file to `src/CMakeLists.txt`.
5. Write CPU gradient tests.
6. Add endpoint modes `align` and `anti_align`.
7. Add optional exact `theta` mode only after cosine mode is validated.
8. Add CUDA struct and enum.
9. Add CUDA host-side conversion.
10. Add CUDA kernel switch case.
11. Add CPU/GPU parity tests.
12. Document the new force in `docs/source/forces.md`.

The stable default I would ship first is:

\[
E=\frac{1}{2}k_c\left[
\frac{(\mathbf{R}_A-\mathbf{R}_B)\cdot(\mathbf{R}_C-\mathbf{R}_B)}
{\|\mathbf{R}_A-\mathbf{R}_B\|\,\|\mathbf{R}_C-\mathbf{R}_B\|}
-\cos\theta_0
\right]^2
\]

with forces distributed over all three groups as derived above. This is meaningful, conservative, CPU/GPU-portable, and avoids the worst angular singularities.

