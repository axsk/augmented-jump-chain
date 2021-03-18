module NN

using Flux
using Plots
using LinearAlgebra
using StatsBase
import Flux.Zygote.@ignore

import Flux.Zygote.hook
import Flux.Zygote.dropgrad
import Flux.Zygote.@adjoint

function sqraloss(f, x, u, h)
    ndim = size(x, 1)
    rr = 0.
    ux = u(x)
    s = 0.
    for i in 1:ndim
        for p in [h, -h]
            @ignore x[i] += p
            uxp = u(x)
            fxp = f(x)
            @ignore x[i] -= p
            r = exp(-1/2 * (uxp - ux))
            rr += r
            s += r * fxp
        end
    end
    s -= rr * f(x)
    return abs(s)
end

function sqraloss_vec1(f, x, u, h)
    n = size(x,1)
    stencil = hcat(zeros(n), diagm(h * ones(n)), diagm(-h * ones(n)))
    testpoints = x .+ stencil
    ut = u(testpoints)
    ft = f(testpoints)
    w = exp.(-1/2 * (ut .- ut[1]))
    w[1] = -sum(w[2:end])
    loss = dot(w, ft)
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

struct NNModel{T,U}
    nn::T
    bndfun::U
end

function NNModel()
    NNModel(
        Chain(Dense(2,10,sigmoid), Dense(10,10,sigmoid), Dense(10,1,sigmoid)),
        triplewellbnd)
end

import Zygote.@showgrad
import Zygote._pullback
#=
function (m::NNModel)(x::AbstractMatrix)
    b = m.bndfun(x)
    c = hook(g->replace(g, nothing=>0), m.nn(x))
    map((b,c) -> isnan(b) ? c : b, b, c)
end

@adjoint triplewellbnd(x::AbstractMatrix) = triplewellbnd(x), d->(nothing, )#(zeros(size(@show d)),)

=#

function (m::NNModel)(x::AbstractMatrix)
    b = m.bndfun(x)
    y = m.nn(x)
    map((b,y) -> isnan(b) ? y : b, b, y)
end

@adjoint function (m::NNModel)(x::AbstractMatrix)
    b = m.bndfun(x)
    y, back = _pullback(m.nn, x)
    y = map((b,c) -> isnan(b) ? y : b, b, y)
    function pb(delta)
        delta = map((b,d) -> isnan(b) ? d : 0., b, delta)
        back(delta)
    end
    y, pb
end

function plot(m::NNModel)
    xs = -3:.1:3
    ys = -2:.1:2
    zs = hcat(([x,y] for x in xs, y in ys)...)
    zs = reshape(m(zs), length(xs), length(ys))'
    heatmap(xs, ys, zs)
    contour!(xs, ys, zs, linewidth=2)


    #heatmap([m([x,y]) for x in -3:.1:3, y in -2:.1:2]')
end

Flux.trainable(m::NNModel) = (m.nn, )

function triplewell(x::AbstractVector)
    x, y = x
    V =  (3/4 * exp(-x^2 - (y-1/3)^2)
        - 3/4 * exp(-x^2 - (y-5/3)^2)
        - 5/4 * exp(-(x-1)^2 - y^2)
        - 5/4 * exp(-(x+1)^2 - y^2)
        + 1/20 * x^4 + 1/20 * (y-1/3)^4)
end


#triplewell(x::Matrix) = map(triplewell, eachcol(x))

#triplewell(x::Matrix) = map(triplewell, eachcol(x))

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


function train(iter=1000, batch=100;
    model = NNModel(
        Chain(Dense(2,10,sigmoid), Dense(10,10,sigmoid), Dense(10,1,sigmoid)),
        triplewellbnd),
    opt = ADAM(0.01),
    h = 0.1)
    ps = Flux.params(model)
    maxloss = 0

    for i in 1:iter
        local losses
        x = rand(2,batch) .* [6, 4] .- [3,2]
        #xs = -3:.2:3
        #ys = -2:.2:2
        #x = hcat(([x,y] for x in xs, y in ys)...)

        l, pb = Flux.pullback(ps) do
            losses = sqraloss_batch(model, x, triplewell, h)
            sum(abs2,losses)
            #sum(sqraloss(model, x, triplewell, h) for x in eachcol(x)) / batch

        end
        #@show losses
        println(l)
        grad = pb(1)
        Flux.Optimise.update!(opt, ps, grad)

        maxloss = max(maximum(losses))
        if i % 10 == 0
        plot(model)# |> display
        Plots.scatter!(x[1,:], x[2,:], legend=false, markersize = losses' / maxloss * 10) |> display
        end
    end
    model
end

end