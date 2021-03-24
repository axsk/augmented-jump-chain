module NN

using Flux
using Plots
using LinearAlgebra
using StatsBase
import Flux.Zygote.@ignore

import Flux.Zygote.hook
import Flux.Zygote.dropgrad
import Flux.Zygote.@adjoint


### plotting

dim(m::Flux.Chain) = size(m.layers[1].W, 2)

function plot(m, bounds, nx=40, ny=30)
    xs = range(bounds[1,:]..., length = nx)
    ys = range(bounds[2,:]..., length = ny)
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

### Boundary Types

abstract type Boundary end

function boundaryfixture(b::Boundary, f)
    function (x)
        a, _, r = getboundary(b, x)
        f(x) .* r .+ a
    end
end

## Crisp Boundary

struct CustomCrisp{T} <: Boundary
    bndcheck::T
end
getboundary(b::CustomCrisp, x) = crispboundary(b.bndcheck, x)

struct RadialCrisp{T<:AbstractVector} <: Boundary
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

## Smooth Boundary

struct RadialExp{T<:AbstractVector} <: Boundary
    a::T
    b::T
    e::Float64
end

getboundary(c::RadialExp, x) = smoothboundary(c.a, c.b, c.e, x)
function smoothboundary(a, b, e, x)
    a = exp.(e * sum(abs2, x .- a, dims=1)) # set a with bnd = 1
    b = exp.(e * sum(abs2, x .- b, dims=1)) # set b with bnd = 0
    r = 1 .- a .- b
    a, b, r
end


### Committor Types

## SQRA Committor
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

## Sampled Committor

struct CommittorSampled{B}
    boundary::B
end

SampledData = Array{<:Tuple{<:Vector,<:Matrix}}

SampledDataTensor = Array{<:Matrix}

# old tuple version, whilst evaluation is only slightly slower, the pullback is 4x slower
function losses(c::CommittorSampled, f, data::D) where D <: SampledData
    fb = boundaryfixture(c.boundary, f)
    losses = [sampleloss(fb, d[1], d[2]) for d in data]'

    # weight losses only inside the solution domain
    xs = hcat((d[1] for d in data)...) :: Matrix{Float64} # TODO find typestable+ad-able
    _, _, r = getboundary(c.boundary, xs)
    losses = losses .* r
end


function losses(c::CommittorSampled, f, data::Array{T, 3}) where T
    f = boundaryfixture(c.boundary, f)
    (i,j,k) = size(data)
    xs = reshape(data, i, j*k)
    ys = reshape(f(xs), j, k)

    losses = abs.(mean(ys[1,:]' .- ys[2:end,:], dims=1))

    _, _, r = getboundary(c.boundary, data[:,1,:])
    losses .* r
end


function sampleloss(f, x, ys)
    abs(mean(f(x) .- f(ys)))
end

function randbox(bounds, n)
    shift = bounds[:,2]
    scale = (bounds[:,2] - bounds[:,1])
    rand(size(bounds, 1), n) .* scale .- shift
end

function mlp(x, in=2, sig=false)
    first = Dense(in, x[1], σ)
    if sig
        last = Dense(x[end], 1, σ)
    else
        last = Dense(x[end], 1)
    end
    Chain(first, [Dense(x[i], x[i+1], σ) for i=1:length(x)-1]..., last)
end


function test_sqra(;hidden=[10,10], h=.1, beta=5., r=.1, samples=10_000, batch=100, plotevery=10, epochs=10)
    model = NN.mlp(hidden)
    c = CommittorSQRA(triplewell, h, beta, RadialCrisp([1,0],[-1,0], r))
    data = randbox(triplewellbox, samples)
    train(model, c, data, bounds=triplewellbox, plotevery=plotevery, batch=batch, epochs=epochs)
end

function test_sampled(;hidden=[10,10], r=.1, samples=1_000, trajs=100, batch=1000, plotevery=10)
    model = mlp(hidden)
    c = CommittorSampled(RadialCrisp([1,0],[-1,0], r))
    xs = randbox(triplewellbox, samples)

    data = [(collect(x), x.+randn(2,trajs) *.1) for x in eachcol(xs)]
    train(model, c, data, bounds=triplewellbox, plotevery=plotevery, batch=batch)
end



function train(model, c, data;
    epochs=1,
    batch=1000,
    bounds=[],
    opt = ADAM(0.01),
    plotevery=1
    )

    ps = Flux.params(model)
    losshist = []

    for i in 1:epochs
        for x in Flux.Data.DataLoader(data, batchsize=batch)
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


            if length(losshist) % plotevery == 0
                maxloss = max(maximum(pointlosses))
                p1 = plot(model, bounds)# |> display
                #Plots.plot()
                if isa(x,Matrix{Float64})
                    p1 = Plots.scatter!(x[1,:], x[2,:], legend=false,
                        marker=((pointlosses' / maxloss).^(1/1) * 10, stroke(0)))
                elseif isa(x, SampledData)
                    for tup in x
                        x, ys = tup
                        Plots.scatter!([x[1]], [x[2]])
                        for y in eachcol(ys)
                            Plots.plot!([x[1],y[1]], [x[2], y[2]], legend=false)
                        end
                    end
                end
                p2 = Plots.plot(losshist, yscale=:log10)
                Plots.plot(p1, p2, layout=@layout([a;b]), size=(800,800)) |> display
            end
        end
    end
    model
end

import Zygote.gradient

eulermaruyamastep(x, force, sigma, dt) = x .+  force * dt .+ sigma * randn(size(x)) * sqrt(dt)

function eulermaruyama(x, potential, sigma, dt, n)
    for i in 1:n
        x = eulermaruyamastep(x, -gradient(potential, x)[1], sigma, dt)
    end
    x
end

struct RandomData{T}
    generator::T
end

function Flux.Data.DataLoader(d::RandomData; batchsize=1)
    [d.generator(batchsize)]
end





using Parameters

@with_kw struct Potential2D
    r = .1
    box = [-3. 3; -2 2]
    emsigma = .1
end

potential(::Potential2D) = triplewell
box(t::Potential2D) = t.box
boundary(t::Potential2D) = RadialCrisp([1.,0],[-1.,0], t.r)
sample(t::Potential2D, n) = randbox(t.box, n)
randdata(t::Potential2D, branches, dt, steps) = RandomData(n->sampletrajectories(t, n, branches, dt=dt, steps=steps))

function sampletrajectories(t::Potential2D, n, m; dt=1, steps=1)
    xs = sample(t, n)
    u = potential(t)
    sigma = t.emsigma
    [(collect(x), hcat([eulermaruyama(x, u, sigma, dt, steps) for i in 1:m]...)) for x in eachcol(xs)]
end

function test(p::Potential2D; hidden=[10,10], samples=100, branches=1, plotevery=1, dt=.1, steps=10, epochs=10)
    model = mlp(hidden)
    c = CommittorSampled(boundary(p))
    #data = sampletrajectories(p, samples, branches, dt=dt, steps=steps)
    data = randdata(p, branches, dt, steps)
    train(model, c, data, bounds=triplewellbox, plotevery=plotevery, batch=samples, epochs=epochs)
end


#=

defaultmodel(c::Committorproblem) = Chain(Dense(dim(c), 10, σ), Dense(10, 10, σ), Dense(10, 1, σ))
sample(c::Committorproblem, n) = samplebox(c.bounds, n)
dim(c::Committorproblem) = size(c.bounds, 1)





dim(c::CommittorSampled) = size(c.bounds, 1)

getboundary(c::CommittorSampled, x) = getboundary(c.boundary, x)

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

=#



export train

# concrete models

#defaultproblem() = TriplewellCrisp()

# Mullerbrown
MullerBrown() = CommittorSQRA(mullerbrown, .1, 5, RadialCrisp([0.6,0], [-0.55,1.4], .1))
#, mullerbrownbox)

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
#TriplewellCrisp() = CommittorCrispBoundary(triplewell, 0.1, 5., triplewellbnd, 2, triplewellbox)

#TriplewellSmooth() = CommittorSmoothBoundary(triplewell, 0.1, 5., [1.,0], [-1.,0], 30., triplewellbox)

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

function triplewellcustombnd(v)
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

function triplewellcustombnd(x::AbstractMatrix)
    triplewellbnd.(eachcol(x))
end
@adjoint triplewellcustombnd(x::AbstractMatrix) = triplewellcustombnd(x), d->(nothing, )#(zeros(size(@show d)),)

# Hyperwell
#=
function hyperproblem(n=2; h=.1, beta=5.)
    a = ones(n)
    b = ones(n)
    b[1] = -1
    CommittorSmoothBoundary(hyperwell, h, beta, a, b, 30.)
end =#

hyperwell(x) = 0.1*sum((x.^2 .-1).^2, dims=1)


end # module
