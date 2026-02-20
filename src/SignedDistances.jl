module SignedDistances

using Base.Threads
using GeometryBasics
using LinearAlgebra

export SignedDistanceMesh, preprocess_mesh, compute_signed_distance, compute_signed_distance!

##################################   Feature Codes & Utilities   ##################################

# feature codes double as tuple indices into TriangleNormals.normals,
# enabling O(1) branchless pseudonormal lookup: normals[feat].
const FEAT_V1 = UInt8(1)     # vertex 1 (a)
const FEAT_V2 = UInt8(2)     # vertex 2 (b)
const FEAT_V3 = UInt8(3)     # vertex 3 (c)
const FEAT_E12 = UInt8(4)    # edge AB  (v1–v2)
const FEAT_E23 = UInt8(5)    # edge BC  (v2–v3)
const FEAT_E31 = UInt8(6)    # edge CA  (v3–v1)
const FEAT_FACE = UInt8(7)   # face interior

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

# stable angle via atan(‖ × ‖, ⋅ ) — avoids division, clamp, and acos instability near 0°/180°
@inline function angle_between(u::Point3{T}, v::Point3{T}) where {T<:AbstractFloat}
    x = u ⋅ v
    y = √(norm²(u × v))
    α = atan(y, x)
    return α::T
end

#######################################   Data Structures   #######################################

# packed triangle vertices (contiguous by BVH leaf order for cache locality)
struct TriangleGeometry{T<:AbstractFloat}
    a::Point3{T}
    ab::Point3{T}
    ac::Point3{T}
end

# all 7 pseudonormals packed per-triangle.
# tuple indices match feature codes for O(1) lookup: normals[feat]
#   [1]=v1  [2]=v2  [3]=v3  [4]=e12  [5]=e23  [6]=e31  [7]=face
struct TriangleNormals{T<:AbstractFloat}
    normals::NTuple{7,Point3{T}}
end

# BVH with SoA layout (each field is a separate array for minimal cache-footprint during pruning)
struct BoundingVolumeHierarchy{T<:AbstractFloat}
    lb_x::Vector{T}
    lb_y::Vector{T}
    lb_z::Vector{T}
    ub_x::Vector{T}
    ub_y::Vector{T}
    ub_z::Vector{T}
    left::Vector{Int32}
    right::Vector{Int32}
    # leaf range start (indexes into packed tri arrays)
    leaf_start::Vector{Int32}
    # leaf size (0 → internal node)
    leaf_sizes::Vector{Int32}
    leaf_capacity::Int32
    num_nodes::Int32
end

# stack element carrying both node id and its already-computed AABB lower bound.
# this avoids recomputing aabb_dist² again when the node is popped.
struct NodeDist{T<:AbstractFloat}
    node::Int32
    dist²::T
end

# Tg: geometry/distance type (Float32 recommended)
# Ts: pseudonormal/sign type (Float64 recommended)
struct SignedDistanceMesh{Tg<:AbstractFloat,Ts<:AbstractFloat}
    tri_geometries::Vector{TriangleGeometry{Tg}}   # packed by BVH leaf order
    tri_normals::Vector{TriangleNormals{Ts}}       # packed by BVH leaf order
    bvh::BoundingVolumeHierarchy{Tg}
    # face_to_packed[f] = packed triangle index for original face id f
    # (used to exploit your “source triangle” hints)
    face_to_packed::Vector{Int32}
    # pre-allocated per-thread traversal stacks (avoid allocations in hot loop)
    stacks::Vector{Vector{NodeDist{Tg}}}
end

#######################################   BVH Construction   #######################################

# partial sort of indices by centroid along axis (build-time only)
function median_split_sort!(
    indices::Vector{Int32},
    slice::AbstractUnitRange{Int},
    centroids::NTuple{3,Vector{Tg}},
    axis::Int
) where {Tg<:AbstractFloat}
    sub_indices = @view indices[slice]
    centroids_axis = centroids[axis]
    mid = length(sub_indices) ÷ 2 + 1
    partialsort!(sub_indices, mid; by=tri_idx -> centroids_axis[tri_idx])
    return indices
end

mutable struct BVHBuilder{T}
    const lb_x::Vector{T}
    const lb_y::Vector{T}
    const lb_z::Vector{T}
    const ub_x::Vector{T}
    const ub_y::Vector{T}
    const ub_z::Vector{T}
    const left::Vector{Int32}
    const right::Vector{Int32}
    const leaf_start::Vector{Int32}
    const leaf_sizes::Vector{Int32}
    const leaf_capacity::Int32
    next_node::Int32
end

function build_node!(
    builder::BVHBuilder{T},
    tri_indices::Vector{Int32}, lo::Int, hi::Int, centroids::NTuple{3,Vector{T}},
    lb_x_t::Vector{T}, lb_y_t::Vector{T}, lb_z_t::Vector{T},
    ub_x_t::Vector{T}, ub_y_t::Vector{T}, ub_z_t::Vector{T},
) where {T}
    node = builder.next_node
    builder.next_node += 1

    # compute node bounds
    min_x = T(Inf)
    min_y = T(Inf)
    min_z = T(Inf)
    max_x = -T(Inf)
    max_y = -T(Inf)
    max_z = -T(Inf)
    @inbounds begin
        for i in lo:hi
            t = tri_indices[i]
            min_x = min(min_x, lb_x_t[t])
            min_y = min(min_y, lb_y_t[t])
            min_z = min(min_z, lb_z_t[t])
            max_x = max(max_x, ub_x_t[t])
            max_y = max(max_y, ub_y_t[t])
            max_z = max(max_z, ub_z_t[t])
        end
        builder.lb_x[node] = min_x
        builder.lb_y[node] = min_y
        builder.lb_z[node] = min_z
        builder.ub_x[node] = max_x
        builder.ub_y[node] = max_y
        builder.ub_z[node] = max_z
    end

    n = hi - lo + 1
    if n <= builder.leaf_capacity
        @inbounds begin
            builder.left[node] = 0
            builder.right[node] = 0
            builder.leaf_start[node] = Int32(lo)
            builder.leaf_sizes[node] = Int32(n)
        end
        return node
    end

    # split axis = longest centroid extent
    centroid_min_x = T(Inf)
    centroid_min_y = T(Inf)
    centroid_min_z = T(Inf)
    centroid_max_x = -T(Inf)
    centroid_max_y = -T(Inf)
    centroid_max_z = -T(Inf)
    @inbounds for i in lo:hi
        t = tri_indices[i]
        centroid_min_x = min(centroid_min_x, centroids[1][t])
        centroid_min_y = min(centroid_min_y, centroids[2][t])
        centroid_min_z = min(centroid_min_z, centroids[3][t])
        centroid_max_x = max(centroid_max_x, centroids[1][t])
        centroid_max_y = max(centroid_max_y, centroids[2][t])
        centroid_max_z = max(centroid_max_z, centroids[3][t])
    end
    spread_x = centroid_max_x - centroid_min_x
    spread_y = centroid_max_y - centroid_min_y
    spread_z = centroid_max_z - centroid_min_z
    (spread_max, axis) = findmax((spread_x, spread_y, spread_z))

    mid = (lo + hi) >>> 1   # (lo + hi) ÷ 2
    # median split via partial sort (skip if all centroids identical along all axes)
    (spread_max > 0) && median_split_sort!(tri_indices, lo:hi, centroids, axis)

    leftnode = build_node!(
        builder, tri_indices, lo, mid, centroids, lb_x_t, lb_y_t, lb_z_t, ub_x_t, ub_y_t, ub_z_t
    )
    rightnode = build_node!(
        builder, tri_indices, mid + 1, hi, centroids, lb_x_t, lb_y_t, lb_z_t, ub_x_t, ub_y_t, ub_z_t
    )
    @inbounds begin
        builder.left[node] = leftnode
        builder.right[node] = rightnode
        builder.leaf_start[node] = 0
        builder.leaf_sizes[node] = 0
    end
    return node
end

function build_bvh(
    centroids::NTuple{3,Vector{Tg}},
    lb_x_t::Vector{Tg}, lb_y_t::Vector{Tg}, lb_z_t::Vector{Tg},
    ub_x_t::Vector{Tg}, ub_y_t::Vector{Tg}, ub_z_t::Vector{Tg};
    leaf_capacity::Int=8
) where {Tg}
    num_faces = length(first(centroids))
    tri_indices = Int32.(1:num_faces)

    max_nodes = 2 * ceil(Int, num_faces / leaf_capacity)
    builder = BVHBuilder{Tg}(  # use Tg for BVHBuilder to avoid allocations
        Vector{Tg}(undef, max_nodes), Vector{Tg}(undef, max_nodes), Vector{Tg}(undef, max_nodes),
        Vector{Tg}(undef, max_nodes), Vector{Tg}(undef, max_nodes), Vector{Tg}(undef, max_nodes),
        Vector{Int32}(undef, max_nodes), Vector{Int32}(undef, max_nodes),
        Vector{Int32}(undef, max_nodes), Vector{Int32}(undef, max_nodes),
        Int32(leaf_capacity), Int32(1)
    )
    build_node!(
        builder, tri_indices, 1, num_faces, centroids, lb_x_t, lb_y_t, lb_z_t, ub_x_t, ub_y_t, ub_z_t
    )

    num_nodes = builder.next_node - 1
    resize!(builder.lb_x, num_nodes)
    resize!(builder.lb_y, num_nodes)
    resize!(builder.lb_z, num_nodes)
    resize!(builder.ub_x, num_nodes)
    resize!(builder.ub_y, num_nodes)
    resize!(builder.ub_z, num_nodes)
    resize!(builder.left, num_nodes)
    resize!(builder.right, num_nodes)
    resize!(builder.leaf_start, num_nodes)
    resize!(builder.leaf_sizes, num_nodes)

    bvh = BoundingVolumeHierarchy{Tg}(
        builder.lb_x, builder.lb_y, builder.lb_z,
        builder.ub_x, builder.ub_y, builder.ub_z,
        builder.left, builder.right, builder.leaf_start, builder.leaf_sizes,
        builder.leaf_capacity, num_nodes
    )
    return (bvh, tri_indices)  # return triangle order for packing
end

######################################   Mesh Preprocessing   ######################################

@inline function edge_key(a::Int32, b::Int32)
    (lo, hi) = minmax(a, b)
    key = (UInt64(lo) << 32) | UInt64(hi)
    return key
end

"""
    preprocess_mesh(vertices, faces; leaf_capacity=8, stack_capacity=256, sign_type=Float64)

Build the acceleration structure for signed-distance queries on a
watertight, consistently-oriented triangle mesh.

- `vertices`:  `3 × num_vertices` matrix of vertex positions (Float32 recommended).
- `faces`:     `3 × num_faces` matrix of 1-based vertex indices.

`sign_type` controls the floating-point type used for *pseudo-normal sign tests*.
Using `Float64` is recommended for robustness of the inside/outside sign.

Returns a `SignedDistanceMesh{Tg,Ts}` ready for `compute_signed_distance!` calls.
"""
function preprocess_mesh(
    mesh::Mesh{3,Float32,GLTriangleFace};
    leaf_capacity::Int=8, sign_type::Type{Ts}=Float64
) where {Ts<:AbstractFloat}
    Tg = Float32
    vertices = GeometryBasics.coordinates(mesh)
    tri_faces = GeometryBasics.faces(mesh)
    faces = NTuple{3,Int32}.(tri_faces)
    num_vertices = length(vertices)
    num_faces = length(faces)

    # face unit normals computed in Ts (Float64 recommended)
    normals = Vector{Point3{Ts}}(undef, num_faces)
    @inbounds for idx_face in eachindex(faces)
        (idx_v1, idx_v2, idx_v3) = faces[idx_face]
        v1 = Point3{Ts}(vertices[idx_v1])
        v2 = Point3{Ts}(vertices[idx_v2])
        v3 = Point3{Ts}(vertices[idx_v3])
        normal = (v2 - v1) × (v3 - v1)
        normals[idx_face] = normalize(normal)
    end

    # face_adjacency[edge, face] = index of the face sharing local `edge` of `face` (0 if boundary)
    face_adjacency = zeros(Int32, 3, num_faces)
    neighbors = Dict{UInt64,Tuple{Int32,Int32}}()
    @inbounds for idx_face in eachindex(faces)
        (idx_v1, idx_v2, idx_v3) = faces[idx_face]
        for (edge, vertex_a, vertex_b) in ((Int32(1), idx_v1, idx_v2), (Int32(2), idx_v2, idx_v3), (Int32(3), idx_v3, idx_v1))
            key = edge_key(vertex_a, vertex_b)
            pair = get(neighbors, key, nothing)
            if pair === nothing
                neighbors[key] = (Int32(idx_face), edge)
            else
                (face_adjacent, edge_common) = pair
                face_adjacency[edge, idx_face] = face_adjacent
                face_adjacency[edge_common, face_adjacent] = Int32(idx_face)
                delete!(neighbors, key)
            end
        end
    end
    @assert isempty(neighbors) "mesh is not watertight: $(length(neighbors)) boundary edges"

    # edge pseudonormals: sum of adjacent unit face normals (unnormalized as only sign matters)
    pns_edge = Matrix{Point3{Ts}}(undef, 3, num_faces)
    for idx_face₁ in eachindex(normals)
        normal₁ = normals[idx_face₁]
        for edge in 1:3
            idx_face₂ = face_adjacency[edge, idx_face₁]
            normal₂ = normals[idx_face₂]
            pns_edge[edge, idx_face₁] = normal₁ + normal₂
        end
    end

    # vertex pseudonormals (angle-weighted, unnormalized) in Ts
    pns_vertex = zeros(Point3{Ts}, num_vertices)
    @inbounds for idx_face in eachindex(faces)
        face = faces[idx_face]
        (idx_v1, idx_v2, idx_v3) = face
        v1 = Point3{Ts}(vertices[idx_v1])
        v2 = Point3{Ts}(vertices[idx_v2])
        v3 = Point3{Ts}(vertices[idx_v3])
        normal = normals[idx_face]

        α1 = angle_between(v2 - v1, v3 - v1)
        α2 = angle_between(v3 - v2, v1 - v2)
        α3 = angle_between(v1 - v3, v2 - v3)

        # intentionally unnormalized: only sign(rvec ⋅ pn) matters for Bærentzen signing
        pns_vertex[idx_v1] += α1 * normal
        pns_vertex[idx_v2] += α2 * normal
        pns_vertex[idx_v3] += α3 * normal
    end

    # build BVH (in Tg)
    lb_x_t = Vector{Tg}(undef, num_faces)
    lb_y_t = Vector{Tg}(undef, num_faces)
    lb_z_t = Vector{Tg}(undef, num_faces)
    ub_x_t = Vector{Tg}(undef, num_faces)
    ub_y_t = Vector{Tg}(undef, num_faces)
    ub_z_t = Vector{Tg}(undef, num_faces)
    centroids_x = Vector{Tg}(undef, num_faces)
    centroids_y = Vector{Tg}(undef, num_faces)
    centroids_z = Vector{Tg}(undef, num_faces)

    @inbounds for idx_face in eachindex(faces)
        face = faces[idx_face]
        (idx_v1, idx_v2, idx_v3) = face
        v1 = vertices[idx_v1]
        v2 = vertices[idx_v2]
        v3 = vertices[idx_v3]
        (x1, y1, z1) = v1
        (x2, y2, z2) = v2
        (x3, y3, z3) = v3

        lb_x_t[idx_face] = min(x1, x2, x3)
        lb_y_t[idx_face] = min(y1, y2, y3)
        lb_z_t[idx_face] = min(z1, z2, z3)
        ub_x_t[idx_face] = max(x1, x2, x3)
        ub_y_t[idx_face] = max(y1, y2, y3)
        ub_z_t[idx_face] = max(z1, z2, z3)

        centroid = (v1 + v2 + v3) / Tg(3)
        centroids_x[idx_face] = centroid[1]
        centroids_y[idx_face] = centroid[2]
        centroids_z[idx_face] = centroid[3]
    end
    centroids = (centroids_x, centroids_y, centroids_z)
    (bvh, tri_order) = build_bvh(
        centroids, lb_x_t, lb_y_t, lb_z_t, ub_x_t, ub_y_t, ub_z_t; leaf_capacity
    )

    # pack triangle geometry & normals contiguously by BVH leaf order
    tri_geometries = Vector{TriangleGeometry{Tg}}(undef, num_faces)
    tri_normals = Vector{TriangleNormals{Ts}}(undef, num_faces)

    # map original face index → packed index (for triangle-hint acceleration)
    face_to_packed = Vector{Int32}(undef, num_faces)

    @inbounds for j in eachindex(faces)
        idx_face = tri_order[j]   # original face index
        face_to_packed[idx_face] = Int32(j)

        (idx_v1, idx_v2, idx_v3) = faces[idx_face]
        v1 = vertices[idx_v1]
        v2 = vertices[idx_v2]
        v3 = vertices[idx_v3]
        tri_geometries[j] = TriangleGeometry{Tg}(v1, v2 - v1, v3 - v1)

        # tuple indices match feature codes for direct normals[feat] lookup
        tri_normals[j] = TriangleNormals{Ts}((
            pns_vertex[idx_v1],      # [1] = FEAT_V1
            pns_vertex[idx_v2],      # [2] = FEAT_V2
            pns_vertex[idx_v3],      # [3] = FEAT_V3
            pns_edge[1, idx_face],   # [4] = FEAT_E12
            pns_edge[2, idx_face],   # [5] = FEAT_E23
            pns_edge[3, idx_face],   # [6] = FEAT_E31
            normals[idx_face],       # [7] = FEAT_FACE
        ))
    end

    # per-thread stacks (pre-allocated once, reused across all queries)
    num_threads = Threads.nthreads()
    tree_height = calculate_tree_height(num_faces, leaf_capacity)
    stack_capacity = 2 * tree_height + 4  # small safety margin
    stacks = [Vector{NodeDist{Tg}}(undef, stack_capacity) for _ in 1:num_threads]
    return SignedDistanceMesh{Tg,Ts}(tri_geometries, tri_normals, bvh, face_to_packed, stacks)
end

function calculate_tree_height(num_faces::Int, leaf_capacity::Int)
    num_leaves = max(ceil(num_faces / leaf_capacity), 1.0)
    tree_height = ceil(Int, log2(num_leaves))
    return tree_height::Int
end

##############################   High-Performance Hot Loop Routines   ##############################

# AABB squared distance with single bound load + branchless clamp
@inline function aabb_dist²(
    point::Point3{Tg}, bvh::BoundingVolumeHierarchy{Tg}, node_id::Int32
) where {Tg}
    (point_x, point_y, point_z) = point
    @inbounds begin
        lb_x = bvh.lb_x[node_id]
        ub_x = bvh.ub_x[node_id]
        lb_y = bvh.lb_y[node_id]
        ub_y = bvh.ub_y[node_id]
        lb_z = bvh.lb_z[node_id]
        ub_z = bvh.ub_z[node_id]
    end
    zer = zero(Tg)
    Δx = max(lb_x - point_x, point_x - ub_x, zer)
    Δy = max(lb_y - point_y, point_y - ub_y, zer)
    Δz = max(lb_z - point_z, point_z - ub_z, zer)
    dist² = Δx * Δx + Δy * Δy + Δz * Δz
    return dist²
end

# exact closest-point-on-triangle (Ericson-style) but returns Δ = p - closest_point.
# This avoids computing and storing closest point, and makes the sign test use Δ directly.
@inline function closest_diff_triangle(p::Point3{Tg}, tg::TriangleGeometry{Tg}) where {Tg}
    a = tg.a
    ab = tg.ab
    ac = tg.ac

    ap = p - a
    d1 = ab ⋅ ap
    d2 = ac ⋅ ap
    if (d1 <= 0) && (d2 <= 0)
        return (norm²(ap), ap, FEAT_V1)
    end

    bp = ap - ab
    d3 = ab ⋅ bp
    d4 = ac ⋅ bp
    if (d3 >= 0) && (d4 <= d3)
        return (norm²(bp), bp, FEAT_V2)
    end

    vc = d1 * d4 - d3 * d2
    if (vc <= 0) && (d1 >= 0) && (d3 <= 0)
        v = d1 / (d1 - d3)   # bary: (1-v, v, 0)
        Δ = ap - v * ab      # p - (a + v*ab)
        return (norm²(Δ), Δ, FEAT_E12)
    end

    cp = ap - ac
    d5 = ab ⋅ cp
    d6 = ac ⋅ cp
    if (d6 >= 0) && (d5 <= d6)
        return (norm²(cp), cp, FEAT_V3)
    end

    vb = d5 * d2 - d1 * d6
    if (vb <= 0) && (d2 >= 0) && (d6 <= 0)
        w = d2 / (d2 - d6)   # bary: (1-w, 0, w)
        Δ = ap - w * ac      # p - (a + w*ac)
        return (norm²(Δ), Δ, FEAT_E31)
    end

    va = d3 * d6 - d5 * d4
    if (va <= 0) && (d4 - d3) >= 0 && (d5 - d6) >= 0
        d43 = d4 - d3
        w = d43 / (d43 + d5 - d6)  # bary: (0, 1-w, w)
        bc = ac - ab
        Δ = bp - w * bc
        return (norm²(Δ), Δ, FEAT_E23)
    end

    denom = inv(va + vb + vc)
    v = vb * denom
    w = vc * denom
    Δ = ap - v * ab - w * ac
    return (norm²(Δ), Δ, FEAT_FACE)
end

######################################   Single-Point Query   ######################################

@inline function signed_distance_point(
    sdm::SignedDistanceMesh{Tg,Ts},
    point::Point3{Tg},
    upper_bound²::Tg,
    stack::Vector{NodeDist{Tg}},
    hint_face::Int32
) where {Tg,Ts}
    dist²_best = upper_bound²
    Δ_best = zero(Point3{Tg})
    feat_best = UInt8(0)
    tri_best = Int32(0)

    tri_geometries = sdm.tri_geometries
    # tighten initial bound using the provided triangle hint (packed index).
    # this is especially effective for near-surface samples.
    @inbounds (dist²_hint, Δ_hint, feat_hint) = closest_diff_triangle(point, tri_geometries[hint_face])
    if dist²_hint < dist²_best
        dist²_best = dist²_hint
        Δ_best = Δ_hint
        feat_best = feat_hint
        tri_best = hint_face
    end

    bvh = sdm.bvh
    stack_top = 1
    dist²_root = aabb_dist²(point, bvh, Int32(1))
    @inbounds stack[1] = NodeDist{Tg}(Int32(1), dist²_root)

    @inbounds while stack_top > 0
        node_dist = stack[stack_top]
        stack_top -= 1

        (node_dist.dist² > dist²_best) && continue

        node = node_dist.node
        num_triangles = bvh.leaf_sizes[node]

        if num_triangles == 0 # internal node
            # compute child AABB bounds once, push with stored distances
            child_l = bvh.left[node]
            child_r = bvh.right[node]

            dist²_l = aabb_dist²(point, bvh, child_l)
            dist²_r = aabb_dist²(point, bvh, child_r)

            # sort so near child is pushed last (popped first)
            if dist²_l > dist²_r
                (child_l, child_r) = (child_r, child_l)
                (dist²_l, dist²_r) = (dist²_r, dist²_l)
            end

            # push far, then near
            if dist²_r <= dist²_best
                stack_top += 1
                stack[stack_top] = NodeDist{Tg}(child_r, dist²_r)
            end
            if dist²_l <= dist²_best
                stack_top += 1
                stack[stack_top] = NodeDist{Tg}(child_l, dist²_l)
            end
        else # leaf: test triangles (data is contiguous in packed arrays)
            leaf_start = bvh.leaf_start[node]
            leaf_end = leaf_start + num_triangles - Int32(1)
            for idx in leaf_start:leaf_end
                (dist², Δ, feat) = closest_diff_triangle(point, tri_geometries[idx])
                if dist² < dist²_best
                    dist²_best = dist²
                    Δ_best = Δ
                    feat_best = feat
                    tri_best = idx
                end
            end
        end
    end

    iszero(tri_best) && error("No triangle found")
    dist = √(dist²_best)
    iszero(dist) && return zero(Tg)

    # O(1) branchless lookup: tuple index == feature code
    @inbounds pseudonormal = sdm.tri_normals[tri_best].normals[feat_best]

    # compute sign in Float64 for robustness (even if geometry is Float32)
    dot64 = Float64(Δ_best[1]) * Float64(pseudonormal[1]) +
            Float64(Δ_best[2]) * Float64(pseudonormal[2]) +
            Float64(Δ_best[3]) * Float64(pseudonormal[3])

    uno = one(Tg)
    sgn = ifelse(dot64 >= 0.0, uno, -uno)
    signed_dist = sgn * dist
    return signed_dist
end

##########################################   Public API   ##########################################

"""
    compute_signed_distance!(out, sdm, points_mat, upper_bounds², hint_faces)

In-place batch signed distance query. Writes results into `out`.
- `out`:            length-n vector to store the output signed distances.
- `sdm`:            a [`SignedDistanceMesh`] built once via `preprocess_mesh`.
- `points_mat`:     `3 × n` matrix of query points (Float32 recommended).
- `upper_bounds²`:  length-n vector of unsigned distance upper bounds per point.
                    Pass `Inf` for any point without a known bound.
- `hint_faces`:     length-n vector of *original* face indices (1-based,
                    matching the input `faces`) for each point.
                    This uses a single exact triangle check to tighten the upper bound before
                    BVH traversal, which can substantially speed up near-surface queries.

Positive = outside, negative = inside.

Notes:
- The unsigned distance is computed in the geometry type `Tg` (Float32).
- The *sign decision* (inner product with angle-weighted pseudo-normal) is computed in Float64.
"""
function compute_signed_distance!(
    out::AbstractVector{Tg},
    sdm::SignedDistanceMesh{Tg,Ts},
    points_mat::StridedMatrix{Tg},
    upper_bounds²::AbstractVector{Tg},
    hint_faces::Vector{Int32}
) where {Tg<:AbstractFloat,Ts<:AbstractFloat}
    @assert size(points_mat, 1) == 3 "points matrix must be 3×n"
    num_points = size(points_mat, 2)
    @assert length(out) == num_points
    @assert length(upper_bounds²) == num_points
    @assert length(hint_faces) == num_points

    stacks = sdm.stacks
    face_to_packed = sdm.face_to_packed
    # Zero-copy reinterpret: treat columns of X as Point3{Tg} without allocation
    points = reinterpret(reshape, Point3{Tg}, points_mat)

    # SAFETY WARNING: :static scheduling is REQUIRED here. threadid() is used to index
    # pre-allocated per-thread stacks. Switching to :dynamic will cause data races.
    Threads.@threads :static for i in 1:num_points
        thread_id = Threads.threadid()
        @inbounds begin
            idx_face = hint_faces[i]
            idx_face_packed = face_to_packed[idx_face]
            out[i] = signed_distance_point(
                sdm, points[i], upper_bounds²[i], stacks[thread_id], idx_face_packed
            )
        end
    end
    return out
end

"""
    compute_signed_distance(sd, X, upper_bounds², hint_faces) → Vector{Tg}

Allocating batch signed distance query. See `compute_signed_distance!`.
"""
function compute_signed_distance(
    sdm::SignedDistanceMesh{Tg,Ts},
    points::StridedMatrix{Tg},
    upper_bounds²::AbstractVector{Tg},
    hint_faces::Vector{Int32}
) where {Tg<:AbstractFloat,Ts<:AbstractFloat}
    num_points = size(points, 2)
    out = Vector{Tg}(undef, num_points)
    compute_signed_distance!(out, sdm, points, upper_bounds², hint_faces)
    return out
end

end # module
