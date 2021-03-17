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

function sqraloss_batch(f, x, u, h)
    n,m = size(x)
    stencil = hcat(zeros(n), diagm(h * ones(n)), diagm(-h * ones(n)))
    testpoints = reshape(x, n, 1, m) .+ stencil |> x->reshape(x, n, m*(2n + 1))
    ut = u(testpoints) |> x->reshape(x, 2*n+1, m)
    ft = f(testpoints) |> x->reshape(x, 2*n+1, m)
    w = exp.(-1/2 * (ut .- ut[1, :]'))
    #w[1,:] = -sum(w[2:end, :], dims=1)
    w = vcat(-sum(w[2:end, :], dims=1), w[2:end,:])
    loss = sum(abs.(sum(w .* ft, dims=1)))
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

function (m::NNModel)(x::AbstractMatrix)

    b = dropgrad(m.bndfun(x))
    c = m.nn(x)

    map((b,c) -> isnan(b) ? c : b, b, c)

end

@adjoint (m::NNModel)(x) = m(x),  d -> begin
    @show typeof(d), size(d)
    f, back = Flux.pullback(m.nn, x)
    @show 0, back(d)
end

plot(m::NNModel) = heatmap([m([x,y]) for x in -3:.1:3, y in -2:.1:2]')

Flux.trainable(m::NNModel) = (m.nn, )

function triplewell(x::AbstractVector)
    x, y = x
    V =  (3/4 * exp(-x^2 - (y-1/3)^2)
        - 3/4 * exp(-x^2 - (y-5/3)^2)
        - 5/4 * exp(-(x-1)^2 - y^2)
        - 5/4 * exp(-(x+1)^2 - y^2)
        + 1/20 * x^4 + 1/20 * (y-1/3)^4)
end


triplewell(x::Matrix) = map(triplewell, eachcol(x))

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
    #@show size(x)
    #hook(x->@show(replace(x, nothing=>0)),map(triplewellbnd, eachcol(x)))
    triplewellbnd.(eachcol(x))
end

#@adjoint triplewellbnd(x) = triplewellbnd(x), d->zero(x)


function train(iter=100, batch=100;
    model = NNModel(
        Chain(Dense(2,10,sigmoid), Dense(10,10,sigmoid), Dense(10,1,sigmoid)),
        triplewellbnd),
    opt = ADAM(0.01),
    h = 0.1)
    ps = Flux.params(model)
    for i in 1:iter
        x = rand(2,batch) .* [2, 2] .- [1,1]
        l, pb = Flux.pullback(ps) do
            sqraloss_batch(model, x, triplewell, h)
            #sum(sqraloss(model, x, triplewell, h) for x in eachcol(x)) / batch

        end
        println(l)
        grad = pb(1)
        Flux.Optimise.update!(opt, ps, grad)

        display(plot(model))
    end
    model
end

end