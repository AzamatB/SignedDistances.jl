module SignedDistances

using Base.Threads
using GeometryBasics
using LinearAlgebra

export SignedDistanceMesh, preprocess_mesh, compute_signed_distance, compute_signed_distance!

##################################   Feature Codes & Utilities   ##################################

# Feature codes double as tuple indices into TriangleNormals.normals,
# enabling O(1) branchless pseudonormal lookup: normals[feat].
const FEAT_V1 = UInt8(1)  # vertex 1 (a)
const FEAT_V2 = UInt8(2)  # vertex 2 (b)
const FEAT_V3 = UInt8(3)  # vertex 3 (c)
const FEAT_E12 = UInt8(4)  # edge AB  (v1–v2)
const FEAT_E23 = UInt8(5)  # edge BC  (v2–v3)
const FEAT_E31 = UInt8(6)  # edge CA  (v3–v1)
const FEAT_FACE = UInt8(7)  # face interior

@inline function norm²(point::Point3{T}) where {T<:AbstractFloat}
    n² = point ⋅ point
    return n²::T
end

@inline function normalize(point::Point3{T}) where {T<:AbstractFloat}
    ε = nextfloat(Float32)
    n² = norm²(point)
    c = (n² > ε) * inv(√(n²))
    point_n = c * point
    return point_n::Point3{T}
end

# Stable angle via atan(‖ × ‖, ⋅ ) — avoids division, clamp, and acos instability near 0°/180°
@inline function angle_between(u::Point3{T}, v::Point3{T}) where {T<:AbstractFloat}
    x = u ⋅ v
    y = √(norm²(u × v))
    α = atan(y, x)
    return α::T
end

#######################################   Data Structures   #######################################

# Packed triangle vertices (contiguous by BVH leaf order for cache locality)
struct TriangleGeometry{T<:AbstractFloat}
    v1::Point3{T}
    v2::Point3{T}
    v3::Point3{T}
end

# All 7 pseudonormals packed per-triangle.
# Tuple indices match feature codes for O(1) lookup: normals[feat]
#   [1]=v1  [2]=v2  [3]=v3  [4]=e12  [5]=e23  [6]=e31  [7]=face
struct TriangleNormals{T<:AbstractFloat}
    normals::NTuple{7,Point3{T}}
end

# BVH with SoA layout (each field is a separate array for minimal cache-footprint during pruning)
struct BVH{T<:AbstractFloat}
    bmin_x::Vector{T}
    bmin_y::Vector{T}
    bmin_z::Vector{T}
    bmax_x::Vector{T}
    bmax_y::Vector{T}
    bmax_z::Vector{T}
    left::Vector{Int32}
    right::Vector{Int32}
    # leaf range start (indexes into packed tri arrays)
    first::Vector{Int32}
    # leaf count (0 → internal node)
    count::Vector{Int32}
    num_nodes::Int32
    leaf_size::Int32
end

# Stack element carrying both node id and its already-computed AABB lower bound.
# This avoids recomputing aabb_dist² again when the node is popped.
struct NodeDist{T<:AbstractFloat}
    node::Int32
    dist²::T
end


# Tg: geometry/distance type (Float32 recommended)
# Ts: pseudonormal/sign type (Float64 recommended)
struct SignedDistanceMesh{Tg<:AbstractFloat,Ts<:AbstractFloat}
    tri_geometries::Vector{TriangleGeometry{Tg}}          # packed by BVH leaf order
    tri_normals::Vector{TriangleNormals{Ts}}   # packed by BVH leaf order
    bvh::BVH{Tg}

    # face_to_packed[f] = packed triangle index for original face id f
    # (used to exploit your “source triangle” hints)
    face_to_packed::Vector{Int32}

    # pre-allocated per-thread traversal stacks (avoid allocations in hot loop)
    stacks::Vector{Vector{NodeDist{Tg}}}
end

#######################################   BVH Construction   #######################################

# Quickselect partition by centroid along axis (build-time only)
function nth_element!(idxs::Vector{Int32}, lo::Int, hi::Int, k::Int,
    axis::Int, cx::Vector, cy::Vector, cz::Vector)
    @inline getc(tri::Int32) = (axis == 1 ? cx[tri] : (axis == 2 ? cy[tri] : cz[tri]))

    while lo < hi
        pivot = idxs[(lo+hi)>>>1]
        pivotv = getc(pivot)

        i = lo
        j = hi
        @inbounds while i <= j
            while getc(idxs[i]) < pivotv
                i += 1
            end
            while getc(idxs[j]) > pivotv
                j -= 1
            end
            if i <= j
                idxs[i], idxs[j] = idxs[j], idxs[i]
                i += 1
                j -= 1
            end
        end

        if k <= j
            hi = j
        elseif k >= i
            lo = i
        else
            return
        end
    end
end

mutable struct BVHBuilder{T}
    bmin_x::Vector{T}
    bmin_y::Vector{T}
    bmin_z::Vector{T}
    bmax_x::Vector{T}
    bmax_y::Vector{T}
    bmax_z::Vector{T}
    left::Vector{Int32}
    right::Vector{Int32}
    first::Vector{Int32}
    count::Vector{Int32}
    next_node::Int32
    leaf_size::Int32
end

function build_node!(builder::BVHBuilder{T},
    idxs::Vector{Int32}, lo::Int, hi::Int,
    tbminx::Vector{T}, tbminy::Vector{T}, tbminz::Vector{T},
    tbmaxx::Vector{T}, tbmaxy::Vector{T}, tbmaxz::Vector{T},
    cx::Vector{T}, cy::Vector{T}, cz::Vector{T}) where {T}

    node = builder.next_node
    builder.next_node += 1

    # Compute node bounds
    minx = T(Inf)
    miny = T(Inf)
    minz = T(Inf)
    maxx = -T(Inf)
    maxy = -T(Inf)
    maxz = -T(Inf)
    @inbounds for i in lo:hi
        t = idxs[i]
        minx = min(minx, tbminx[t])
        miny = min(miny, tbminy[t])
        minz = min(minz, tbminz[t])
        maxx = max(maxx, tbmaxx[t])
        maxy = max(maxy, tbmaxy[t])
        maxz = max(maxz, tbmaxz[t])
    end
    @inbounds begin
        builder.bmin_x[node] = minx
        builder.bmin_y[node] = miny
        builder.bmin_z[node] = minz
        builder.bmax_x[node] = maxx
        builder.bmax_y[node] = maxy
        builder.bmax_z[node] = maxz
    end

    n = hi - lo + 1
    if n <= builder.leaf_size
        @inbounds begin
            builder.left[node] = 0
            builder.right[node] = 0
            builder.first[node] = Int32(lo)
            builder.count[node] = Int32(n)
        end
        return node
    end

    # Split axis = longest centroid extent
    cminx = T(Inf)
    cminy = T(Inf)
    cminz = T(Inf)
    cmaxx = -T(Inf)
    cmaxy = -T(Inf)
    cmaxz = -T(Inf)
    @inbounds for i in lo:hi
        t = idxs[i]
        cminx = min(cminx, cx[t])
        cminy = min(cminy, cy[t])
        cminz = min(cminz, cz[t])
        cmaxx = max(cmaxx, cx[t])
        cmaxy = max(cmaxy, cy[t])
        cmaxz = max(cmaxz, cz[t])
    end
    ex = cmaxx - cminx
    ey = cmaxy - cminy
    ez = cmaxz - cminz
    axis = (ex >= ey && ex >= ez) ? 1 : ((ey >= ez) ? 2 : 3)

    mid = (lo + hi) >>> 1
    # Median split via quickselect (skip if all centroids identical along all axes)
    if ex > 0 || ey > 0 || ez > 0
        nth_element!(idxs, lo, hi, mid, axis, cx, cy, cz)
    end

    leftnode = build_node!(builder, idxs, lo, mid,
        tbminx, tbminy, tbminz, tbmaxx, tbmaxy, tbmaxz, cx, cy, cz)
    rightnode = build_node!(builder, idxs, mid + 1, hi,
        tbminx, tbminy, tbminz, tbmaxx, tbmaxy, tbmaxz, cx, cy, cz)

    @inbounds begin
        builder.left[node] = leftnode
        builder.right[node] = rightnode
        builder.first[node] = 0
        builder.count[node] = 0
    end

    return node
end

function build_bvh(::Type{T}, tbminx, tbminy, tbminz, tbmaxx, tbmaxy, tbmaxz, cx, cy, cz;
    leaf_size::Int=8) where {T}
    ntri = length(cx)
    tri = Vector{Int32}(undef, ntri)
    @inbounds for i in 1:ntri
        tri[i] = Int32(i)
    end

    max_nodes = 2 * ntri - 1
    builder = BVHBuilder{T}(
        Vector{T}(undef, max_nodes), Vector{T}(undef, max_nodes), Vector{T}(undef, max_nodes),
        Vector{T}(undef, max_nodes), Vector{T}(undef, max_nodes), Vector{T}(undef, max_nodes),
        Vector{Int32}(undef, max_nodes), Vector{Int32}(undef, max_nodes),
        Vector{Int32}(undef, max_nodes), Vector{Int32}(undef, max_nodes),
        Int32(1), Int32(leaf_size),
    )

    build_node!(builder, tri, 1, ntri, tbminx, tbminy, tbminz, tbmaxx, tbmaxy, tbmaxz, cx, cy, cz)

    num_nodes = builder.next_node - 1
    return BVH{T}(
        builder.bmin_x, builder.bmin_y, builder.bmin_z,
        builder.bmax_x, builder.bmax_y, builder.bmax_z,
        builder.left, builder.right, builder.first, builder.count,
        num_nodes, builder.leaf_size
    ), tri  # return tri order for packing
end

######################################   Mesh Preprocessing   ######################################

@inline edge_key(a::Int32, b::Int32) = a < b ? (UInt64(a) << 32) | UInt64(b) : (UInt64(b) << 32) | UInt64(a)

"""
    preprocess_mesh(V, F; leaf_size=8, stack_capacity=256, sign_type=Float64)

Build the acceleration structure for signed-distance queries on a
watertight, consistently-oriented triangle mesh.

- `V`: `3 × nV` matrix of vertex positions (Float32 recommended).
- `F`: `3 × nF` matrix of 1-based vertex indices.

`sign_type` controls the floating-point type used for *pseudo-normal sign tests*.
Using `Float64` is recommended for robustness of the inside/outside sign.

Returns a `SignedDistanceMesh{Tg,Ts}` ready for `compute_signed_distance!` calls.
"""
function preprocess_mesh(Vmat::StridedMatrix{Tg}, Fmat::StridedMatrix{<:Integer};
    leaf_size::Int=8, stack_capacity::Int=256, sign_type::Type{Ts}=Float64
) where {Tg<:AbstractFloat,Ts<:AbstractFloat}

    @assert size(Vmat, 1) == 3 "V must be 3×nV"
    @assert size(Fmat, 1) == 3 "F must be 3×nF"

    nV = size(Vmat, 2)
    nF = size(Fmat, 2)

    # Vertices as Vector{Point3{Tg}}
    V = reinterpret(Point3{Tg}, vec(Vmat)) |> collect

    # Faces as Int32 matrix
    F = Matrix{Int32}(undef, 3, nF)
    @inbounds for f in 1:nF
        F[1, f] = Int32(Fmat[1, f])
        F[2, f] = Int32(Fmat[2, f])
        F[3, f] = Int32(Fmat[3, f])
    end

    # ── Face normals (unit) computed in Ts (Float64 recommended) ──
    faceN = Vector{Point3{Ts}}(undef, nF)
    @inbounds for f in 1:nF
        a32, b32, c32 = V[F[1, f]], V[F[2, f]], V[F[3, f]]
        a = Point3{Ts}(Ts(a32[1]), Ts(a32[2]), Ts(a32[3]))
        b = Point3{Ts}(Ts(b32[1]), Ts(b32[2]), Ts(b32[3]))
        c = Point3{Ts}(Ts(c32[1]), Ts(c32[2]), Ts(c32[3]))
        faceN[f] = normalize((b - a) × (c - a))
    end

    # ── Edge adjacency ──
    # adj[e, f] = index of the face sharing local edge e of face f (0 if boundary)
    adj = zeros(Int32, 3, nF)
    dict = Dict{UInt64,Tuple{Int32,Int32}}()
    @inbounds for f in 1:nF
        i1, i2, i3 = F[1, f], F[2, f], F[3, f]
        for (e, va, vb) in ((Int32(1), i1, i2), (Int32(2), i2, i3), (Int32(3), i3, i1))
            ek = edge_key(va, vb)
            if haskey(dict, ek)
                (f2, e2) = dict[ek]
                adj[e, f] = f2
                adj[e2, f2] = Int32(f)
                delete!(dict, ek)
            else
                dict[ek] = (Int32(f), e)
            end
        end
    end

    # ── Edge pseudonormals: sum of adjacent unit face normals (unnormalized — only sign matters) ──
    edgePN = Vector{Point3{Ts}}(undef, 3 * nF)
    @inbounds for f in 1:nF
        nf = faceN[f]
        for e in 1:3
            nb = adj[e, f]
            edgePN[3*(f-1)+e] = nb == 0 ? nf : (nf + faceN[nb])
        end
    end

    # ── Vertex pseudonormals (angle-weighted, unnormalized) in Ts ──
    # Accumulate into scalar arrays to avoid Point3 mutation overhead
    accx = zeros(Ts, nV)
    accy = zeros(Ts, nV)
    accz = zeros(Ts, nV)
    @inbounds for f in 1:nF
        i1, i2, i3 = F[1, f], F[2, f], F[3, f]
        a32, b32, c32 = V[i1], V[i2], V[i3]
        a = Point3{Ts}(Ts(a32[1]), Ts(a32[2]), Ts(a32[3]))
        b = Point3{Ts}(Ts(b32[1]), Ts(b32[2]), Ts(b32[3]))
        c = Point3{Ts}(Ts(c32[1]), Ts(c32[2]), Ts(c32[3]))
        nf = faceN[f]

        α1 = angle_between(b - a, c - a)
        α2 = angle_between(c - b, a - b)
        α3 = angle_between(a - c, b - c)

        accx[i1] += α1 * nf[1]
        accy[i1] += α1 * nf[2]
        accz[i1] += α1 * nf[3]
        accx[i2] += α2 * nf[1]
        accy[i2] += α2 * nf[2]
        accz[i2] += α2 * nf[3]
        accx[i3] += α3 * nf[1]
        accy[i3] += α3 * nf[2]
        accz[i3] += α3 * nf[3]
    end

    vertPN = Vector{Point3{Ts}}(undef, nV)
    @inbounds for v in 1:nV
        vertPN[v] = Point3{Ts}(accx[v], accy[v], accz[v])
        # Intentionally unnormalized: only sign(rvec ⋅ pn) matters for Bærentzen signing
    end

    # ── Build BVH (in Tg) ──
    tbminx = Vector{Tg}(undef, nF)
    tbminy = Vector{Tg}(undef, nF)
    tbminz = Vector{Tg}(undef, nF)
    tbmaxx = Vector{Tg}(undef, nF)
    tbmaxy = Vector{Tg}(undef, nF)
    tbmaxz = Vector{Tg}(undef, nF)
    cx = Vector{Tg}(undef, nF)
    cy = Vector{Tg}(undef, nF)
    cz = Vector{Tg}(undef, nF)
    @inbounds for f in 1:nF
        a, b, c = V[F[1, f]], V[F[2, f]], V[F[3, f]]
        tbminx[f] = min(a[1], b[1], c[1])
        tbminy[f] = min(a[2], b[2], c[2])
        tbminz[f] = min(a[3], b[3], c[3])
        tbmaxx[f] = max(a[1], b[1], c[1])
        tbmaxy[f] = max(a[2], b[2], c[2])
        tbmaxz[f] = max(a[3], b[3], c[3])
        cc = (a + b + c) / Tg(3)
        cx[f] = cc[1]
        cy[f] = cc[2]
        cz[f] = cc[3]
    end
    bvh, tri_order = build_bvh(Tg, tbminx, tbminy, tbminz, tbmaxx, tbmaxy, tbmaxz, cx, cy, cz; leaf_size=leaf_size)

    # ── Pack triangle geometry & normals contiguously by BVH leaf order ──
    tri_geometries = Vector{TriangleGeometry{Tg}}(undef, nF)
    tri_normals = Vector{TriangleNormals{Ts}}(undef, nF)

    # Map original face index → packed index (for triangle-hint acceleration)
    face_to_packed = Vector{Int32}(undef, nF)

    @inbounds for j in 1:nF
        f = tri_order[j]   # original face index
        face_to_packed[f] = Int32(j)

        i1, i2, i3 = F[1, f], F[2, f], F[3, f]

        tri_geometries[j] = TriangleGeometry{Tg}(V[i1], V[i2], V[i3])

        # Tuple indices match feature codes for direct normals[feat] lookup
        tri_normals[j] = TriangleNormals{Ts}((
            vertPN[i1],              # [1] = FEAT_V1
            vertPN[i2],              # [2] = FEAT_V2
            vertPN[i3],              # [3] = FEAT_V3
            edgePN[3*(f-1)+1],       # [4] = FEAT_E12
            edgePN[3*(f-1)+2],       # [5] = FEAT_E23
            edgePN[3*(f-1)+3],       # [6] = FEAT_E31
            faceN[f],                # [7] = FEAT_FACE
        ))
    end

    # Per-thread stacks (pre-allocated once, reused across all queries)
    stacks = [Vector{NodeDist{Tg}}(undef, stack_capacity) for _ in 1:Threads.nthreads()]

    return SignedDistanceMesh{Tg,Ts}(tri_geometries, tri_normals, bvh, face_to_packed, stacks)
end

# ============================================================================
# 5. High-Performance Hot Loop Routines
# ============================================================================

# AABB squared distance with single bound load + branchless clamp
@inline function aabb_dist²(p::Point3{Tg}, bvh::BVH{Tg}, node::Int32) where {Tg}
    ni = Int(node)
    px, py, pz = p
    @inbounds begin
        minx = bvh.bmin_x[ni]
        maxx = bvh.bmax_x[ni]
        miny = bvh.bmin_y[ni]
        maxy = bvh.bmax_y[ni]
        minz = bvh.bmin_z[ni]
        maxz = bvh.bmax_z[ni]
    end
    dx = max(max(minx - px, px - maxx), zero(Tg))
    dy = max(max(miny - py, py - maxy), zero(Tg))
    dz = max(max(minz - pz, pz - maxz), zero(Tg))
    return dx * dx + dy * dy + dz * dz
end

# Exact closest-point-on-triangle (Ericson-style) but returns diff = p - closest_point.
# This avoids computing/storing cp and makes the sign test use diff directly.
@inline function closest_diff_triangle(p::Point3{Tg}, tg::TriangleGeometry{Tg}) where {Tg}
    a = tg.v1
    b = tg.v2
    c = tg.v3
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = ab ⋅ ap
    d2 = ac ⋅ ap
    if d1 <= 0 && d2 <= 0
        return norm²(ap), ap, FEAT_V1
    end

    bp = p - b
    d3 = ab ⋅ bp
    d4 = ac ⋅ bp
    if d3 >= 0 && d4 <= d3
        return norm²(bp), bp, FEAT_V2
    end

    vc = d1 * d4 - d3 * d2
    if vc <= 0 && d1 >= 0 && d3 <= 0
        v = d1 / (d1 - d3)              # bary: (1-v, v, 0)
        diff = ap - v * ab              # p - (a + v*ab)
        return norm²(diff), diff, FEAT_E12
    end

    cpv = p - c
    d5 = ab ⋅ cpv
    d6 = ac ⋅ cpv
    if d6 >= 0 && d5 <= d6
        return norm²(cpv), cpv, FEAT_V3
    end

    vb = d5 * d2 - d1 * d6
    if vb <= 0 && d2 >= 0 && d6 <= 0
        w = d2 / (d2 - d6)              # bary: (1-w, 0, w)
        diff = ap - w * ac
        return norm²(diff), diff, FEAT_E31
    end

    va = d3 * d6 - d5 * d4
    if va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))  # bary: (0, 1-w, w)
        bc = c - b
        diff = bp - w * bc
        return norm²(diff), diff, FEAT_E23
    end

    denom = inv(va + vb + vc)
    v = vb * denom
    w = vc * denom
    diff = ap - v * ab - w * ac
    return norm²(diff), diff, FEAT_FACE
end

# ============================================================================
# 6. Single-Point Query
# ============================================================================

@inline function signed_distance_point(sd::SignedDistanceMesh{Tg,Ts},
    p::Point3{Tg}, ub::Tg, stack::Vector{NodeDist{Tg}}, hint_packed::Int32
) where {Tg,Ts}

    bvh = sd.bvh

    # Epsilon margin absorbs float precision issues at the boundary of the upper bound,
    # reducing expensive fallback-to-Inf frequency.
    dist²_best = isfinite(ub) ? ub * ub * (one(Tg) + Tg(1e-4)) + eps(Tg) : Tg(Inf)
    best_diff = zero(Point3{Tg})
    best_feat = UInt8(0)
    best_tri = Int32(0)

    # Optional: tighten initial bound using the provided triangle hint (packed index).
    # This is especially effective for near-surface samples.
    if hint_packed != 0
        @inbounds begin
            d2h, diffh, feath = closest_diff_triangle(p, sd.tri_geometries[hint_packed])
        end
        if d2h < dist²_best
            dist²_best = d2h
            best_diff = diffh
            best_feat = feath
            best_tri = hint_packed
        end
    end

    sp = 1
    @inbounds stack[1] = NodeDist{Tg}(Int32(1), aabb_dist²(p, bvh, Int32(1)))

    while sp > 0
        @inbounds nd = stack[sp]
        sp -= 1

        if nd.dist² > dist²_best
            continue
        end

        node = nd.node
        @inbounds cnt = bvh.count[node]

        if cnt != 0
            # ── Leaf: test triangles (data is contiguous in packed arrays) ──
            @inbounds first = bvh.first[node]
            j0 = Int(first)
            j1 = j0 + Int(cnt) - 1
            @inbounds for j in j0:j1
                dist², diff, feat = closest_diff_triangle(p, sd.tri_geometries[j])
                if dist² < dist²_best
                    dist²_best = dist²
                    best_diff = diff
                    best_feat = feat
                    best_tri = Int32(j)
                end
            end
        else
            # ── Internal: compute child AABB bounds once, push with stored distances ──
            @inbounds l = bvh.left[node]
            @inbounds r = bvh.right[node]

            dl = aabb_dist²(p, bvh, l)
            dr = aabb_dist²(p, bvh, r)

            pushL = (dl <= dist²_best)
            pushR = (dr <= dist²_best)

            if pushL | pushR
                # One resize check per internal node (instead of per push)
                needed = sp + (pushL ? 1 : 0) + (pushR ? 1 : 0)
                if needed > length(stack)
                    resize!(stack, max(needed, 2 * length(stack)))
                end

                # Push farther first so nearer is popped next (depth-first)
                if dl < dr
                    if pushR
                        sp += 1
                        @inbounds stack[sp] = NodeDist{Tg}(r, dr)
                    end
                    if pushL
                        sp += 1
                        @inbounds stack[sp] = NodeDist{Tg}(l, dl)
                    end
                else
                    if pushL
                        sp += 1
                        @inbounds stack[sp] = NodeDist{Tg}(l, dl)
                    end
                    if pushR
                        sp += 1
                        @inbounds stack[sp] = NodeDist{Tg}(r, dr)
                    end
                end
            end
        end
    end

    # Robustness fallback: if upper bound was too tight, retry with unbounded search
    if best_tri == 0 && isfinite(ub)
        return signed_distance_point(sd, p, Tg(Inf), stack, hint_packed)
    end

    d = sqrt(best2)
    if d == zero(Tg)
        return zero(Tg)
    end

    # O(1) branchless lookup: tuple index == feature code
    @inbounds pn = sd.tri_normals[best_tri].normals[Int(best_feat)]

    # Compute sign in Float64 for robustness (even if geometry is Float32)
    dot64 = Float64(best_diff[1]) * Float64(pn[1]) +
            Float64(best_diff[2]) * Float64(pn[2]) +
            Float64(best_diff[3]) * Float64(pn[3])

    s = dot64 >= 0.0 ? one(Tg) : -one(Tg)
    return d * s
end

# ============================================================================
# 7. Public API
# ============================================================================

"""
    compute_signed_distance!(out, sd, X, upper_bounds; threaded=true, hint_faces=nothing)

In-place batch signed distance query. Writes results into `out`.

- `sd`:           a [`SignedDistanceMesh`] built once via `preprocess_mesh`.
- `X`:            `3 × n` matrix of query points (Float32 recommended).
- `upper_bounds`: length-n vector of unsigned distance upper bounds per point.
                  Pass `Inf` for any point without a known bound.
- `hint_faces`:   (optional) length-n vector of *original* face indices (1-based,
                  matching the input `F`) for each point; use 0 for "no hint".
                  This uses a single exact triangle check to tighten the upper bound
                  before BVH traversal, which can substantially speed up near-surface queries.

Positive = outside, negative = inside.

Notes:
- The unsigned distance is computed in the geometry type `Tg` (typically Float32).
- The *sign decision* (inner product with angle-weighted pseudo-normal) is computed in Float64.
"""
function compute_signed_distance!(out::AbstractVector{Tg},
    sd::SignedDistanceMesh{Tg,Ts},
    X::StridedMatrix{Tg},
    upper_bounds::AbstractVector{Tg};
    threaded::Bool=true,
    hint_faces::Union{Nothing,AbstractVector{<:Integer}}=nothing
) where {Tg<:AbstractFloat,Ts<:AbstractFloat}

    @assert size(X, 1) == 3 "X must be 3×n"
    n = size(X, 2)
    @assert length(out) == n
    @assert length(upper_bounds) == n
    if hint_faces !== nothing
        @assert length(hint_faces) == n
    end

    # Zero-copy reinterpret: treat columns of X as Point3{Tg} without allocation
    Xv = reinterpret(Point3{Tg}, vec(X))

    Threads.@threads :static for i in 1:n
        tid = Threads.threadid()
        h = Int32(hint_faces[i])
        hp = h == 0 ? Int32(0) : (@inbounds sd.face_to_packed[h])
        @inbounds out[i] = signed_distance_point(sd, Xv[i], upper_bounds[i], sd.stacks[tid], hp)
    end

    return out
end

"""
    compute_signed_distance(sd, X, upper_bounds; threaded=true, hint_faces=nothing) → Vector{Tg}

Allocating batch signed distance query. See `compute_signed_distance!`.
"""
function compute_signed_distance(sd::SignedDistanceMesh{Tg,Ts},
    X::StridedMatrix{Tg},
    upper_bounds::AbstractVector{Tg};
    threaded::Bool=true,
    hint_faces::Union{Nothing,AbstractVector{<:Integer}}=nothing
) where {Tg<:AbstractFloat,Ts<:AbstractFloat}

    out = Vector{Tg}(undef, size(X, 2))
    compute_signed_distance!(out, sd, X, upper_bounds; threaded=threaded, hint_faces=hint_faces)
    return out
end

end # module
