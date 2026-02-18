module SignedDistances

using Base.Threads
using GeometryBasics: Point3
using LinearAlgebra

export SignedDistanceMesh, preprocess_mesh, compute_signed_distance, compute_signed_distance!

#################################   Feature Codes and Utilities   #################################

# feature codes double as tuple indices into TriangleNormals.normals
const FEAT_V1 = UInt8(1)   # vertex 1 (a)
const FEAT_V2 = UInt8(2)   # vertex 2 (b)
const FEAT_V3 = UInt8(3)   # vertex 3 (c)
const FEAT_E12 = UInt8(4)  # edge AB (v0-v1)
const FEAT_E23 = UInt8(5)  # edge BC (v1-v2)
const FEAT_E31 = UInt8(6)  # edge CA (v2-v0)
const FEAT_FACE = UInt8(7) # face interior

@inline function norm²(point::Point3{T}) where {T<:AbstractFloat}
    magnitude = point ⋅ point
    return magnitude::T
end

@inline function normalize_safe(point::Point3{T}) where {T<:AbstractFloat}
    ε = nextfloat(Float32)
    magnitude = norm²(point)
    c = (magnitude > ε) * inv(√(magnitude))
    point_normalized = c * point
    return point_normalized::Point3{T}
end

# angle_between is stable near 0 and 180 degrees
@inline function angle_between(u::Point3{T}, v::Point3{T}) where {T<:AbstractFloat}
    x = u ⋅ v
    y = √(norm²(u × v))
    α = atan(y, x)
    return α::T
end

#######################################   Data Structures   #######################################

# packed triangle vertices stay contiguous by BVH leaf order for cache locality
struct TriangleGeometry{T<:AbstractFloat}
    v0::Point3{T}
    v1::Point3{T}
    v2::Point3{T}
end

# tuple indices match FEAT_* for O(1) pseudonormal lookup: normals[feat]
struct TriangleNormals{T<:AbstractFloat}
    normals::NTuple{7,Point3{T}}
end

# Struct-of-Arrays (SoA) Bounding Volume Hierarchy (BVH) layout keeps pruning reads compact
struct BVH{T<:AbstractFloat}
    bminx::Vector{T}
    bminy::Vector{T}
    bminz::Vector{T}
    bmaxx::Vector{T}
    bmaxy::Vector{T}
    bmaxz::Vector{T}
    left::Vector{Int32}
    right::Vector{Int32}
    first::Vector{Int32} # leaf range start into packed triangle arrays
    count::Vector{Int32} # leaf count, 0 indicates internal node
    num_nodes::Int32
    leaf_size::Int32
end

# node id plus precomputed AABB lower bound avoids recomputing when popped
struct NodeDistance{T<:AbstractFloat}
    node::Int32
    dist²::T
end

# geometry and distance type is Tg, pseudonormal sign type is Ts
struct SignedDistanceMesh{Tg<:AbstractFloat,Ts<:AbstractFloat}
    triangle_geometry::Vector{TriangleGeometry{Tg}}
    triangle_normals::Vector{TriangleNormals{Ts}}
    bvh::BVH{Tg}
    face_to_packed::Vector{Int32}
    stacks::Vector{Vector{NodeDistance{Tg}}}
end

#######################   BVH Construction via Quickselect   #######################

# partition indices by centroid along one axis for kth selection
function nth_element!(
    idxs::Vector{Int32},
    lo::Int,
    hi::Int,
    k::Int,
    axis::Int,
    cx::Vector,
    cy::Vector,
    cz::Vector
)
    @inline function centroid_get(tri::Int32)
        if axis == 1
            return cx[tri]
        elseif axis == 2
            return cy[tri]
        end
        return cz[tri]
    end

    while lo < hi
        pivot = idxs[(lo+hi)>>>1]
        pivotv = centroid_get(pivot)

        i = lo
        j = hi
        @inbounds while i <= j
            while centroid_get(idxs[i]) < pivotv
                i += 1
            end
            while centroid_get(idxs[j]) > pivotv
                j -= 1
            end
            if i <= j
                (idxs[i], idxs[j]) = (idxs[j], idxs[i])
                i += 1
                j -= 1
            end
        end

        if k <= j
            hi = j
        elseif k >= i
            lo = i
        else
            return nothing
        end
    end
    return nothing
end

mutable struct BVHBuilder{T<:AbstractFloat}
    bminx::Vector{T}
    bminy::Vector{T}
    bminz::Vector{T}
    bmaxx::Vector{T}
    bmaxy::Vector{T}
    bmaxz::Vector{T}
    left::Vector{Int32}
    right::Vector{Int32}
    first::Vector{Int32}
    count::Vector{Int32}
    next_node::Int32
    leaf_size::Int32
end

function build_node!(
    builder::BVHBuilder{T},
    idxs::Vector{Int32},
    lo::Int,
    hi::Int,
    tbminx::Vector{T},
    tbminy::Vector{T},
    tbminz::Vector{T},
    tbmaxx::Vector{T},
    tbmaxy::Vector{T},
    tbmaxz::Vector{T},
    cx::Vector{T},
    cy::Vector{T},
    cz::Vector{T}
) where {T<:AbstractFloat}
    node = builder.next_node
    builder.next_node += 1

    # load node bounds once to maximize scalar reuse in this loop
    minx = T(Inf)
    miny = T(Inf)
    minz = T(Inf)
    maxx = -T(Inf)
    maxy = -T(Inf)
    maxz = -T(Inf)
    @inbounds for i in lo:hi
        tri = idxs[i]
        minx = min(minx, tbminx[tri])
        miny = min(miny, tbminy[tri])
        minz = min(minz, tbminz[tri])
        maxx = max(maxx, tbmaxx[tri])
        maxy = max(maxy, tbmaxy[tri])
        maxz = max(maxz, tbmaxz[tri])
    end

    @inbounds begin
        builder.bminx[node] = minx
        builder.bminy[node] = miny
        builder.bminz[node] = minz
        builder.bmaxx[node] = maxx
        builder.bmaxy[node] = maxy
        builder.bmaxz[node] = maxz
    end

    tri_count = hi - lo + 1
    if tri_count <= builder.leaf_size
        @inbounds begin
            builder.left[node] = 0
            builder.right[node] = 0
            builder.first[node] = Int32(lo)
            builder.count[node] = Int32(tri_count)
        end
        return node
    end

    # split on longest centroid extent to reduce overlap
    cminx = T(Inf)
    cminy = T(Inf)
    cminz = T(Inf)
    cmaxx = -T(Inf)
    cmaxy = -T(Inf)
    cmaxz = -T(Inf)
    @inbounds for i in lo:hi
        tri = idxs[i]
        cminx = min(cminx, cx[tri])
        cminy = min(cminy, cy[tri])
        cminz = min(cminz, cz[tri])
        cmaxx = max(cmaxx, cx[tri])
        cmaxy = max(cmaxy, cy[tri])
        cmaxz = max(cmaxz, cz[tri])
    end

    ex = cmaxx - cminx
    ey = cmaxy - cminy
    ez = cmaxz - cminz
    axis = 3
    if ex >= ey && ex >= ez
        axis = 1
    elseif ey >= ez
        axis = 2
    end

    mid = (lo + hi) >>> 1
    if ex > 0 || ey > 0 || ez > 0
        nth_element!(idxs, lo, hi, mid, axis, cx, cy, cz)
    end

    left_node = build_node!(
        builder,
        idxs,
        lo,
        mid,
        tbminx,
        tbminy,
        tbminz,
        tbmaxx,
        tbmaxy,
        tbmaxz,
        cx,
        cy,
        cz
    )
    right_node = build_node!(
        builder,
        idxs,
        mid + 1,
        hi,
        tbminx,
        tbminy,
        tbminz,
        tbmaxx,
        tbmaxy,
        tbmaxz,
        cx,
        cy,
        cz
    )

    @inbounds begin
        builder.left[node] = left_node
        builder.right[node] = right_node
        builder.first[node] = 0
        builder.count[node] = 0
    end

    return node
end

function build_bvh(
    ::Type{T},
    tbminx,
    tbminy,
    tbminz,
    tbmaxx,
    tbmaxy,
    tbmaxz,
    cx,
    cy,
    cz;
    leaf_size::Int=8
) where {T<:AbstractFloat}
    tri_count = length(cx)
    tri_idxs = Vector{Int32}(undef, tri_count)
    @inbounds for i in eachindex(tri_idxs)
        tri_idxs[i] = Int32(i)
    end

    max_nodes = 2 * tri_count - 1
    builder = BVHBuilder{T}(
        Vector{T}(undef, max_nodes),
        Vector{T}(undef, max_nodes),
        Vector{T}(undef, max_nodes),
        Vector{T}(undef, max_nodes),
        Vector{T}(undef, max_nodes),
        Vector{T}(undef, max_nodes),
        Vector{Int32}(undef, max_nodes),
        Vector{Int32}(undef, max_nodes),
        Vector{Int32}(undef, max_nodes),
        Vector{Int32}(undef, max_nodes),
        Int32(1),
        Int32(leaf_size)
    )

    build_node!(
        builder,
        tri_idxs,
        1,
        tri_count,
        tbminx,
        tbminy,
        tbminz,
        tbmaxx,
        tbmaxy,
        tbmaxz,
        cx,
        cy,
        cz
    )

    num_nodes = builder.next_node - 1
    bvh = BVH{T}(
        builder.bminx,
        builder.bminy,
        builder.bminz,
        builder.bmaxx,
        builder.bmaxy,
        builder.bmaxz,
        builder.left,
        builder.right,
        builder.first,
        builder.count,
        num_nodes,
        builder.leaf_size
    )
    return (bvh, tri_idxs)
end

#############################   Mesh Preprocessing   ##############################

@inline function edge_key(a::Int32, b::Int32)
    if a < b
        return (UInt64(a) << 32) | UInt64(b)
    end
    return (UInt64(b) << 32) | UInt64(a)
end

"""
    preprocess_mesh(V, F; leaf_size=8, stack_capacity=256, sign_type=Float64)

build the acceleration structure for signed-distance queries on a
watertight, consistently-oriented triangle mesh

- `V`: `3 × nV` matrix of vertex positions (Float32 recommended)
- `F`: `3 × nF` matrix of 1-based vertex indices

`sign_type` controls the floating-point type used for pseudo-normal sign tests
using `Float64` is recommended for inside/outside robustness

returns a `SignedDistanceMesh{Tg,Ts}` ready for `compute_signed_distance!` calls
"""
function preprocess_mesh(
    Vmat::StridedMatrix{Tg},
    Fmat::StridedMatrix{<:Integer};
    leaf_size::Int=8,
    stack_capacity::Int=256,
    sign_type::Type{Ts}=Float64
) where {Tg<:AbstractFloat,Ts<:AbstractFloat}
    @assert size(Vmat, 1) == 3 "V must be 3×nV"
    @assert size(Fmat, 1) == 3 "F must be 3×nF"

    n_vertex = size(Vmat, 2)
    n_face = size(Fmat, 2)

    vertices = reinterpret(Point3{Tg}, vec(Vmat)) |> collect

    faces = Matrix{Int32}(undef, 3, n_face)
    @inbounds for f in axes(faces, 2)
        faces[1, f] = Int32(Fmat[1, f])
        faces[2, f] = Int32(Fmat[2, f])
        faces[3, f] = Int32(Fmat[3, f])
    end

    face_normals = Vector{Point3{Ts}}(undef, n_face)
    @inbounds for f in axes(faces, 2)
        (a32, b32, c32) = (vertices[faces[1, f]], vertices[faces[2, f]], vertices[faces[3, f]])
        a = Point3{Ts}(Ts(a32[1]), Ts(a32[2]), Ts(a32[3]))
        b = Point3{Ts}(Ts(b32[1]), Ts(b32[2]), Ts(b32[3]))
        c = Point3{Ts}(Ts(c32[1]), Ts(c32[2]), Ts(c32[3]))
        normal_face = cross(b - a, c - a)
        face_normals[f] = normalize_safe(Point3{Ts}(normal_face[1], normal_face[2], normal_face[3]))
    end

    # local edge adjacency drives edge pseudonormal accumulation
    adj = zeros(Int32, 3, n_face)
    edge_to_face = Dict{UInt64,Tuple{Int32,Int32}}()
    @inbounds for f in axes(faces, 2)
        (i1, i2, i3) = (faces[1, f], faces[2, f], faces[3, f])
        for (e, v_a, v_b) in ((Int32(1), i1, i2), (Int32(2), i2, i3), (Int32(3), i3, i1))
            key = edge_key(v_a, v_b)
            if haskey(edge_to_face, key)
                (f2, e2) = edge_to_face[key]
                adj[e, f] = f2
                adj[e2, f2] = Int32(f)
                delete!(edge_to_face, key)
            else
                edge_to_face[key] = (Int32(f), e)
            end
        end
    end

    edge_pseudonormals = Vector{Point3{Ts}}(undef, 3 * n_face)
    @inbounds for f in axes(faces, 2)
        normal_face = face_normals[f]
        for e in axes(adj, 1)
            normal_nb_idx = adj[e, f]
            edge_idx = 3 * (f - 1) + e
            if normal_nb_idx == 0
                edge_pseudonormals[edge_idx] = normal_face
            else
                normal_sum = normal_face + face_normals[normal_nb_idx]
                edge_pseudonormals[edge_idx] = Point3{Ts}(
                    normal_sum[1],
                    normal_sum[2],
                    normal_sum[3]
                )
            end
        end
    end

    accx = zeros(Ts, n_vertex)
    accy = zeros(Ts, n_vertex)
    accz = zeros(Ts, n_vertex)
    @inbounds for f in axes(faces, 2)
        (i1, i2, i3) = (faces[1, f], faces[2, f], faces[3, f])
        (a32, b32, c32) = (vertices[i1], vertices[i2], vertices[i3])
        a = Point3{Ts}(Ts(a32[1]), Ts(a32[2]), Ts(a32[3]))
        b = Point3{Ts}(Ts(b32[1]), Ts(b32[2]), Ts(b32[3]))
        c = Point3{Ts}(Ts(c32[1]), Ts(c32[2]), Ts(c32[3]))
        normal_face = face_normals[f]

        α1 = angle_between(b - a, c - a)
        α2 = angle_between(c - b, a - b)
        α3 = angle_between(a - c, b - c)

        accx[i1] += α1 * normal_face[1]
        accy[i1] += α1 * normal_face[2]
        accz[i1] += α1 * normal_face[3]
        accx[i2] += α2 * normal_face[1]
        accy[i2] += α2 * normal_face[2]
        accz[i2] += α2 * normal_face[3]
        accx[i3] += α3 * normal_face[1]
        accy[i3] += α3 * normal_face[2]
        accz[i3] += α3 * normal_face[3]
    end

    vertex_pseudonormals = Vector{Point3{Ts}}(undef, n_vertex)
    @inbounds for v in eachindex(vertex_pseudonormals)
        vertex_pseudonormals[v] = Point3{Ts}(accx[v], accy[v], accz[v])
    end

    tbminx = Vector{Tg}(undef, n_face)
    tbminy = Vector{Tg}(undef, n_face)
    tbminz = Vector{Tg}(undef, n_face)
    tbmaxx = Vector{Tg}(undef, n_face)
    tbmaxy = Vector{Tg}(undef, n_face)
    tbmaxz = Vector{Tg}(undef, n_face)
    cx = Vector{Tg}(undef, n_face)
    cy = Vector{Tg}(undef, n_face)
    cz = Vector{Tg}(undef, n_face)
    @inbounds for f in axes(faces, 2)
        (a, b, c) = (vertices[faces[1, f]], vertices[faces[2, f]], vertices[faces[3, f]])
        tbminx[f] = min(a[1], b[1], c[1])
        tbminy[f] = min(a[2], b[2], c[2])
        tbminz[f] = min(a[3], b[3], c[3])
        tbmaxx[f] = max(a[1], b[1], c[1])
        tbmaxy[f] = max(a[2], b[2], c[2])
        tbmaxz[f] = max(a[3], b[3], c[3])
        centroid = (a + b + c) / Tg(3)
        cx[f] = centroid[1]
        cy[f] = centroid[2]
        cz[f] = centroid[3]
    end
    (bvh, tri_order) = build_bvh(
        Tg,
        tbminx,
        tbminy,
        tbminz,
        tbmaxx,
        tbmaxy,
        tbmaxz,
        cx,
        cy,
        cz;
        leaf_size
    )

    triangle_geometry = Vector{TriangleGeometry{Tg}}(undef, n_face)
    triangle_normals = Vector{TriangleNormals{Ts}}(undef, n_face)
    face_to_packed = Vector{Int32}(undef, n_face)

    @inbounds for packed_idx in eachindex(tri_order)
        face_idx = tri_order[packed_idx]
        face_to_packed[face_idx] = Int32(packed_idx)

        (i1, i2, i3) = (faces[1, face_idx], faces[2, face_idx], faces[3, face_idx])
        triangle_geometry[packed_idx] = TriangleGeometry{Tg}(
            vertices[i1],
            vertices[i2],
            vertices[i3]
        )

        edge_base = 3 * (face_idx - 1)
        triangle_normals[packed_idx] = TriangleNormals{Ts}((
            vertex_pseudonormals[i1],
            vertex_pseudonormals[i2],
            vertex_pseudonormals[i3],
            edge_pseudonormals[edge_base+1],
            edge_pseudonormals[edge_base+2],
            edge_pseudonormals[edge_base+3],
            face_normals[face_idx]
        ))
    end

    stacks = [Vector{NodeDistance{Tg}}(undef, stack_capacity) for _ in 1:Threads.nthreads()]
    return SignedDistanceMesh{Tg,Ts}(
        triangle_geometry,
        triangle_normals,
        bvh,
        face_to_packed,
        stacks
    )
end

#########################   Query Hot-Loop Routines   #########################

@inline function aabb_dist²(p::Point3{Tg}, bvh::BVH{Tg}, node::Int32) where {Tg}
    node_idx = Int(node)
    (px, py, pz) = p
    @inbounds begin
        minx = bvh.bminx[node_idx]
        maxx = bvh.bmaxx[node_idx]
        miny = bvh.bminy[node_idx]
        maxy = bvh.bmaxy[node_idx]
        minz = bvh.bminz[node_idx]
        maxz = bvh.bmaxz[node_idx]
    end
    dx = max(max(minx - px, px - maxx), zero(Tg))
    dy = max(max(miny - py, py - maxy), zero(Tg))
    dz = max(max(minz - pz, pz - maxz), zero(Tg))
    return dx * dx + dy * dy + dz * dz
end

# returns squared distance, diff=(p-closest), and FEAT_* code
@inline function closest_diff_triangle(p::Point3{Tg}, tri::TriangleGeometry{Tg}) where {Tg}
    (a, b, c) = (tri.v0, tri.v1, tri.v2)
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = dot(ab, ap)
    d2 = dot(ac, ap)
    if d1 <= 0 && d2 <= 0
        return (norm²(ap), ap, FEAT_V1)
    end

    bp = p - b
    d3 = dot(ab, bp)
    d4 = dot(ac, bp)
    if d3 >= 0 && d4 <= d3
        return (norm²(bp), bp, FEAT_V2)
    end

    vc = d1 * d4 - d3 * d2
    if vc <= 0 && d1 >= 0 && d3 <= 0
        v = d1 / (d1 - d3)
        diff = ap - v * ab
        return (norm²(diff), diff, FEAT_E12)
    end

    cpv = p - c
    d5 = dot(ab, cpv)
    d6 = dot(ac, cpv)
    if d6 >= 0 && d5 <= d6
        return (norm²(cpv), cpv, FEAT_V3)
    end

    vb = d5 * d2 - d1 * d6
    if vb <= 0 && d2 >= 0 && d6 <= 0
        w = d2 / (d2 - d6)
        diff = ap - w * ac
        return (norm²(diff), diff, FEAT_E31)
    end

    va = d3 * d6 - d5 * d4
    if va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        bc = c - b
        diff = bp - w * bc
        return (norm²(diff), diff, FEAT_E23)
    end

    denom = inv(va + vb + vc)
    v = vb * denom
    w = vc * denom
    diff = ap - v * ab - w * ac
    return (norm²(diff), diff, FEAT_FACE)
end

##############################   Single-Point Query   ##############################

@inline function signed_distance_point(
    sd::SignedDistanceMesh{Tg,Ts},
    p::Point3{Tg},
    ub::Tg,
    stack::Vector{NodeDistance{Tg}},
    hint_packed::Int32
) where {Tg,Ts}
    bvh = sd.bvh
    best² = isfinite(ub) ? ub * ub * (one(Tg) + Tg(1.0e-4)) + eps(Tg) : Tg(Inf)
    best_diff = p - p
    best_feat = UInt8(0)
    best_tri = Int32(0)

    if hint_packed != 0
        @inbounds begin
            (d²_hint, diff_hint, feat_hint) = closest_diff_triangle(
                p,
                sd.triangle_geometry[hint_packed]
            )
        end
        if d²_hint < best²
            best² = d²_hint
            best_diff = diff_hint
            best_feat = feat_hint
            best_tri = hint_packed
        end
    end

    sp = 1
    @inbounds stack[1] = NodeDistance{Tg}(Int32(1), aabb_dist²(p, bvh, Int32(1)))

    while sp > 0
        @inbounds nd = stack[sp]
        sp -= 1

        if nd.dist² > best²
            continue
        end

        node = nd.node
        @inbounds tri_count = bvh.count[node]

        if tri_count != 0
            @inbounds first = bvh.first[node]
            j0 = Int(first)
            j1 = j0 + Int(tri_count) - 1
            @inbounds for j in j0:j1
                (d², diff, feat) = closest_diff_triangle(p, sd.triangle_geometry[j])
                if d² < best²
                    best² = d²
                    best_diff = diff
                    best_feat = feat
                    best_tri = Int32(j)
                end
            end
        else
            @inbounds left = bvh.left[node]
            @inbounds right = bvh.right[node]

            dist²_left = aabb_dist²(p, bvh, left)
            dist²_right = aabb_dist²(p, bvh, right)
            push_left = dist²_left <= best²
            push_right = dist²_right <= best²

            if push_left | push_right
                needed = sp + Int(push_left) + Int(push_right)
                if needed > length(stack)
                    resize!(stack, max(needed, 2 * length(stack)))
                end

                # push farther first so the nearer node is popped next
                if dist²_left < dist²_right
                    if push_right
                        sp += 1
                        @inbounds stack[sp] = NodeDistance{Tg}(right, dist²_right)
                    end
                    if push_left
                        sp += 1
                        @inbounds stack[sp] = NodeDistance{Tg}(left, dist²_left)
                    end
                else
                    if push_left
                        sp += 1
                        @inbounds stack[sp] = NodeDistance{Tg}(left, dist²_left)
                    end
                    if push_right
                        sp += 1
                        @inbounds stack[sp] = NodeDistance{Tg}(right, dist²_right)
                    end
                end
            end
        end
    end

    (best_tri == 0 && isfinite(ub)) && return signed_distance_point(
        sd,
        p,
        Tg(Inf),
        stack,
        hint_packed
    )

    d = √(best²)
    (d == zero(Tg)) && return zero(Tg)

    @inbounds pn = sd.triangle_normals[best_tri].normals[Int(best_feat)]
    dot64 = Float64(best_diff[1]) * Float64(pn[1]) +
            Float64(best_diff[2]) * Float64(pn[2]) +
            Float64(best_diff[3]) * Float64(pn[3])
    sign = dot64 >= 0.0 ? one(Tg) : -one(Tg)
    return d * sign
end

###################################   Public API   ###################################

"""
    compute_signed_distance!(out, sd, X, upper_bounds; threaded=true, hint_faces=nothing)

in-place batch signed distance query and write results into `out`

- `sd`: a `SignedDistanceMesh` built via `preprocess_mesh`
- `X`: `3 × n` query point matrix
- `upper_bounds`: length-n vector of unsigned upper bounds, use `Inf` if unknown
- `hint_faces`: optional length-n vector of original face indices, use 0 for no hint

positive values are outside and negative values are inside
"""
function compute_signed_distance!(
    out::AbstractVector{Tg},
    sd::SignedDistanceMesh{Tg,Ts},
    X::StridedMatrix{Tg},
    upper_bounds::AbstractVector{Tg};
    threaded::Bool=true,
    hint_faces::Union{Nothing,AbstractVector{<:Integer}}=nothing
) where {Tg<:AbstractFloat,Ts<:AbstractFloat}
    @assert size(X, 1) == 3 "X must be 3×n"
    n_point = size(X, 2)
    @assert length(out) == n_point
    @assert length(upper_bounds) == n_point
    if hint_faces !== nothing
        @assert length(hint_faces) == n_point
    end
    X_vec = reinterpret(Point3{Tg}, vec(X))
    Threads.@threads :static for i in eachindex(out)
        tid = Threads.threadid()
        @inbounds out[i] = signed_distance_point(
            sd,
            X_vec[i],
            upper_bounds[i],
            sd.stacks[tid],
            Int32(0)
        )
    end
    return out
end

"""
    compute_signed_distance(sd, X, upper_bounds; threaded=true, hint_faces=nothing)

allocating batch signed distance query, see `compute_signed_distance!`
"""
function compute_signed_distance(
    sd::SignedDistanceMesh{Tg,Ts},
    X::StridedMatrix{Tg},
    upper_bounds::AbstractVector{Tg};
    threaded::Bool=true,
    hint_faces::Union{Nothing,AbstractVector{<:Integer}}=nothing
) where {Tg<:AbstractFloat,Ts<:AbstractFloat}
    out = Vector{Tg}(undef, size(X, 2))
    compute_signed_distance!(out, sd, X, upper_bounds; threaded, hint_faces)
    return out
end

end # module
