module NN

using Flux
using Plots

import Flux.Zygote.@ignore

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

struct NNModel
    nn
    bndfun
end

function (m::NNModel)(x)
    b = m.bndfun(x)
    if isnan(b)
        m.nn(x)[1]
    else
        b
    end
end

plot(m::NNModel) = heatmap([m([x,y]) for x in -3:.1:3, y in -2:.1:2]')

Flux.trainable(m::NNModel) = (m.nn, )

function triplewell(x)
    x, y = x
    V =  (3/4 * exp(-x^2 - (y-1/3)^2)
        - 3/4 * exp(-x^2 - (y-5/3)^2)
        - 5/4 * exp(-(x-1)^2 - y^2)
        - 5/4 * exp(-(x+1)^2 - y^2)
        + 1/20 * x^4 + 1/20 * (y-1/3)^4)
end

function triplewellbnd(x)
    x, y = x
    if abs(y) < 0.2
        if 0.8 < x < 1.2
            return 1.
        elseif -.8 > x > -1.2
            return 0
        end
    end
    return NaN
end

using Parameters
using StatsBase

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
            sum(sqraloss(model, x, triplewell, h) for x in eachcol(x)) / batch
        end
        println(l)
        grad = pb(1)
        Flux.Optimise.update!(opt, ps, grad)

        display(plot(model))
    end
    model
end

end