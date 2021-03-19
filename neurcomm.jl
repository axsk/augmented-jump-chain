module NN

using Flux
using Plots
using LinearAlgebra
using StatsBase
import Flux.Zygote.@ignore

import Flux.Zygote.hook
import Flux.Zygote.dropgrad
import Flux.Zygote.@adjoint


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


dim(m::Flux.Chain) = size(m.layers[1].W, 2)

function plot(m)
    xs = -3:.1:3
    ys = -2:.1:2
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

###############


hyperwell(x) = 0.1*sum((x.^2 .-1).^2, dims=1)

abstract type Committorproblem end

using Parameters

@with_kw struct CommittorSmoothBoundary{T} <: Committorproblem
    potential::T
    h::Float64 = .1
    beta::Float64 = 5
    a::Vector{Float64} 
    b::Vector{Float64}
    exp::Float64 = 30
end

dim(c::CommittorSmoothBoundary) = length(c.a)

@with_kw struct CommittorCrispBoundary{T,U} <: Committorproblem
    potential::T
    h::Float64 = .1
    beta::Float64 = 5
    bndcheck::U # function mapping a state to 1, 0 or anything else to encode for A, B or the interior
    dim::Int 
end

dim(c::CommittorCrispBoundary) = c.dim

CommittorCrispBoundary() = CommittorCrispBoundary(triplewell, 0.1, 5., triplewellbnd, 2)


triplewellproplem() = CommittorSmoothBoundary(triplewell, 0.1, 5., [1.,0], [-1.,0], 30.)

function hyperproblem(n=2; h=.1, beta=5.)
    a = ones(n)
    b = ones(n)
    b[1] = -1
    CommittorSmoothBoundary(hyperwell, h, beta, a, b, 30.)
end

defaultmodel(c::Committorproblem) = Chain(Dense(dim(c), 10, σ), Dense(10, 10, σ), Dense(10, 1, σ))

function sample(c::Committorproblem, n)
    if dim(c) == 2
        x = rand(2,n) .* [6, 4] .- [3,2]
    else
        x = rand(dim(c),n) .* 4 .- 2
    end
end

import Zygote.@showgrad

function getboundary(c::CommittorSmoothBoundary, x)
    a = exp.(-c.exp * sum(abs2, x .- c.a, dims=1)) # set a with bnd = 1
    b = exp.(-c.exp * sum(abs2, x .- c.b, dims=1)) # set b with bnd = 0
    r = 1 .- a .- b
    a, b, r
end

@adjoint getboundary(c, x) = getboundary(c, x), delta->(nothing, nothing)

function getboundary(c::CommittorCrispBoundary, x)
    bnd = map(c.bndcheck, eachcol(x))'
    a = bnd .== 1
    b = bnd .== 0
    r = 1 .- a .- b
    a, b, r
end

#@adjoint getboundary(c::CommittorCrispBoundary, x) = getboundary(c, x), delta->(nothing, nothing)


function boundaryfixture(c::Committorproblem, f)
    # given a function, return the function with fixed boundary values
    function (x)
        a, b, r = getboundary(c, x)
        f(x) .* r .+ a
    end
end

function losses(c::Committorproblem, f, x)

    fb = boundaryfixture(c, f)

    losses = sqraloss_batch(fb, x, c.potential, c.h, beta=c.beta)

    # weight losses only inside the solution domain
    _, _, r = getboundary(c, x)
    losses = losses .* r
    losses
end

function train(batch=100, epochs=1000;
        c = triplewellproplem(),
        model = defaultmodel(c),
        opt = ADAM(0.01),
        plotevery=Inf)


    ps = Flux.params(model)
    losshist = []

    for i in 1:epochs
        #x = rand(2,batch) .* [6, 4] .- [3,2]

        x = sample(c, batch)

        #xs = -3:.2:3
        #ys = -2:.2:2
        #x = hcat(([x,y] for x in xs, y in ys)...)

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
            plot(model)# |> display
            p1 = Plots.scatter!(x[1,:], x[2,:], legend=false, 
                marker=((pointlosses' / maxloss).^(1/1) * 10, stroke(0)))
            p2 = Plots.plot(losshist, yscale=:log10)
            Plots.plot(p1, p2, layout=@layout([a;b]), size=(800,800)) |> display
        end
    end
    model
end

export train


end # module