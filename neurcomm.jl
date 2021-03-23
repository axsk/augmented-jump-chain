module NN

using Flux
using Plots
using LinearAlgebra
using StatsBase
import Flux.Zygote.@ignore

import Flux.Zygote.hook
import Flux.Zygote.dropgrad
import Flux.Zygote.@adjoint


### learning core

function sqraloss_batch(f, x, u, h; beta=5)
    n,m = size(x)
    stencil = dropgrad(hcat(zeros(n), diagm(h * ones(n)), diagm(-h * ones(n))))
    testpoints = reshape(x, n, 1, m) .+ stencil |> x->reshape(x, n, m*(2n + 1))
    ut = u(testpoints) |> x->reshape(x, 2*n+1, m)
    ft = f(testpoints) |> x->reshape(x, 2*n+1, m)
    w = exp.(-1/2 * beta * (ut .- ut[1, :]'))
    #w[1,:] = -sum(w[2:end, :], dims=1)
    w = vcat(-sum(w[2:end, :], dims=1), w[2:end,:])
    loss = abs.(sum(w .* ft, dims=1))
end

function sampleloss(f, x, ys)
    abs(mean(f(x) .- f(ys)))
end



function train(batch=100, epochs=1000;
    c = defaultproblem(),
    model = defaultmodel(c),
    opt = ADAM(0.01),
    plotevery=Inf)


    ps = Flux.params(model)
    losshist = []

    for i in 1:epochs
        x = sample(c, batch)

        local pointlosses

        l, pb = Flux.pullback(ps) do
            pointlosses = losses(c, model, x)
            sum(abs2, pointlosses) / size(x, 2)
        end
        #@show losses
        push!(losshist, l)
        println(l)
        grad = pb(1)
        Flux.Optimise.update!(opt, ps, grad)


        if i % plotevery == 0
            maxloss = max(maximum(pointlosses))
            p1 = plot(model, c)# |> display
            #p1 = Plots.scatter!(x[1,:], x[2,:], legend=false,
            #    marker=((pointlosses' / maxloss).^(1/1) * 10, stroke(0)))
            p2 = Plots.plot(losshist, yscale=:log10)
            Plots.plot(p1, p2, layout=@layout([a;b]), size=(800,800)) |> display
        end
    end
    model
end

### plotting

dim(m::Flux.Chain) = size(m.layers[1].W, 2)

function plot(m, c, nx=40, ny=30)
    xs = range(c.bounds[1,:]..., length = nx)
    ys = range(c.bounds[2,:]..., length = ny)
    zs = zeros(dim(m), length(xs) * length(ys))

    i=1
    for y in ys
        for x in xs
            zs[1,i] = x
            zs[2,i] = y
            i+=1
        end
    end

    zs = reshape(m(zs), length(xs), length(ys))'
    heatmap(xs, ys, zs)
    contour!(xs, ys, zs, linewidth=2)
end

### Models


abstract type Committorproblem end
# implements defaultmodel, losses, sample

defaultmodel(c::Committorproblem) = Chain(Dense(dim(c), 10, σ), Dense(10, 10, σ), Dense(10, 1, σ))

function boundaryfixture(c::Committorproblem, f)
    # given a function, return the function with fixed boundary values
    function (x)
        a, b, r = getboundary(c, x)
        f(x) .* r .+ a
    end
end

# since boundary sometimes poses differentiation problems
#@adjoint getboundary(c, x) = getboundary(c, x), delta->(nothing, nothing)

function losses(c::Committorproblem, f, x)

    fb = boundaryfixture(c, f)

    losses = sqraloss_batch(fb, x, c.potential, c.h, beta=c.beta)

    # weight losses only inside the solution domain
    _, _, r = getboundary(c, x)
    losses = losses .* r
    losses
end

sample(c::Committorproblem, n) = samplebox(c.bounds, n)
dim(c::Committorproblem) = size(c.bounds, 1)

function samplebox(bounds, n)
    shift = bounds[:,2]
    scale = (bounds[:,2] - bounds[:,1])
    rand(size(bounds, 1), n) .* scale .- shift
end

### concrete models

using Parameters

# committor problem with shoothed boundaries around points a and b
@with_kw struct CommittorSmoothBoundary{T} <: Committorproblem
    potential::T
    h::Float64 = .1
    beta::Float64 = 5
    a::Vector{Float64}
    b::Vector{Float64}
    exp::Float64 = 30
    bounds::Matrix{Float64}
end

getboundary(c::CommittorSmoothBoundary, x) = smoothboundary(c.a, c.b, x)
function smoothboundary(a, b, x)
    a = exp.(-c.exp * sum(abs2, x .- c.a, dims=1)) # set a with bnd = 1
    b = exp.(-c.exp * sum(abs2, x .- c.b, dims=1)) # set b with bnd = 0
    r = 1 .- a .- b
    a, b, r
end

# committor problem with variable potential and crisp boundary assignment
@with_kw struct CommittorCrispBoundary{T,U} <: Committorproblem
    potential::T
    h::Float64 = .1
    beta::Float64 = 5
    bndcheck::U # function mapping a state to 1, 0 or anything else to encode for A, B or the interior
    dim::Int
    bounds::Matrix{Float64}
end

getboundary(c::CommittorCrispBoundary, x) = crispboundary(c.bndcheck, x)


### Boundary Types

abstract type Boundary end

function boundaryfixture(b::Boundary, f)
    function (x)
        a, b, r = getboundary(b, x)
        f(x) .* r .+ a
    end
end


struct CustomCrisp{T}
    bndcheck::T
end
getboundary(b::CustomCrisp, x) = crispboundary(b.bndcheck, x)

struct RadialCrisp{T<:AbstractVector}
    a::T
    b::T
    r::Float64
end
getboundary(c::RadialCrisp, x) = crispboundary(x->bndcheck(c, x), x)

function bndcheck(c::RadialCrisp, x::AbstractVector)
    if sum(abs2, c.a - x) <= c.r^2
        1
    elseif sum(abs2, c.b - x) <= c.r^2
        0
    else
        -1
    end
end

function crispboundary(bndcheck, x)
    bnd = map(bndcheck, eachcol(x))'
    a = bnd .== 1
    b = bnd .== 0
    r = 1 .- a .- b
    a, b, r
end
@adjoint crispboundary(bndcheck, x) = crispboundary(bndcheck, x), delta->(nothing, nothing)


### Committor Types

### SQRA Committor
struct CommittorSQRA{F, B}
    potential::F
    h::Float64
    beta::Float64
    boundary::B
end

function losses(c::CommittorSQRA, f, x)

    fb = boundaryfixture(c.boundary, f)

    losses = sqraloss_batch(fb, x, c.potential, c.h, beta=c.beta)

    # weight losses only inside the solution domain
    _, _, r = getboundary(c.boundary, x)
    losses = losses .* r
end

### Sampled Committor

struct CommittorSampled{B}
    boundary::T
end

dim(c::CommittorSampled) = size(c.bounds, 1)

getboundary(c::CommittorSampled, x) = getboundary(c.boundary, x)
function losses(c::CommittorSampled, f, data)
    fb = boundaryfixture(c, f)
    [sampleloss(fb, d[1], d[2]) for d in data]
end

function sample(c::CommittorSampled, n)
    d = []
    for i in 1:n
        x = samplebox(c.bounds, 1)[:,1]
        ys = x .+ randn(dim(c), 20) / 10
        push!(d, (x,ys))
    end
    d
end

function test()
    #boundary = Crispboundary(NN.triplewellbnd)#
    boundary = RadialCrisp([-.5, 0], [.5, 0], .1)
    data=[[1,2]]
    sp = CommittorSampled(boundary, data, [-1 1; -1 1])
    train(c=sp, plotevery=10)
end




export train

# concrete models

defaultproblem() = TriplewellCrisp()

# Mullerbrown
MullerBrown() = CommittorSmoothBoundary(mullerbrown, a=[0.6,0], b=[-0.55,1.4], mullerbrownbox)

const mullerbrownbox = [-1.5 1; -.5 2]
function mullerbrown(x::AbstractMatrix)
    A = (-200, -100, -170, 15)
    a = (-1, -1, -6.5, .7)
    b = (0, 0, 11, .6)
    c = (-10, -10, -6.5, .7)
    x0 = (1, 0, -.5, -1)
    y0 = (0, .5, 1.5, 1)

    x, y = x[1,:], x[2,:]
    sum(@. A[k] * exp(a[k]*(x-x0[k])^2 + b[k]*(x-x0[k])*(y-y0[k]) + c[k]*(y-y0[k])^2) for k=1:4)
end

mullerbrown(x::AbstractVector) = mullerbrown(reshape(x, (2,1)))[1]
mullerbrown(x, y) = mullerbrown([x,y])

## Triplewell
TriplewellCrisp() = CommittorCrispBoundary(triplewell, 0.1, 5., triplewellbnd, 2, triplewellbox)

TriplewellSmooth() = CommittorSmoothBoundary(triplewell, 0.1, 5., [1.,0], [-1.,0], 30., triplewellbox)

const triplewellbox = [-3. 3; -2 2]

function triplewell(x::AbstractVector)
    x, y = x
    V =  (3/4 * exp(-x^2 - (y-1/3)^2)
        - 3/4 * exp(-x^2 - (y-5/3)^2)
        - 5/4 * exp(-(x-1)^2 - y^2)
        - 5/4 * exp(-(x+1)^2 - y^2)
        + 1/20 * x^4 + 1/20 * (y-1/3)^4)
end

function triplewell(x::Matrix)
    x, y = x[1,:], x[2,:]
    V = @. (3/4 * exp(-x^2 - (y-1/3)^2)
        - 3/4 * exp(-x^2 - (y-5/3)^2)
        - 5/4 * exp(-(x-1)^2 - y^2)
        - 5/4 * exp(-(x+1)^2 - y^2)
        + 1/20 * x^4 + 1/20 * (y-1/3)^4)
end

function triplewellbnd(v)
    x, y = v
    if abs(y) < 0.2
        if 0.8 < x < 1.2
            return 1.
        elseif -.8 > x > -1.2
            return 0.
        end
    end
    return NaN
end

function triplewellbnd(x::AbstractMatrix)
    triplewellbnd.(eachcol(x))
end
@adjoint triplewellbnd(x::AbstractMatrix) = triplewellbnd(x), d->(nothing, )#(zeros(size(@show d)),)

# Hyperwell

function hyperproblem(n=2; h=.1, beta=5.)
    a = ones(n)
    b = ones(n)
    b[1] = -1
    CommittorSmoothBoundary(hyperwell, h, beta, a, b, 30.)
end

hyperwell(x) = 0.1*sum((x.^2 .-1).^2, dims=1)


end # module