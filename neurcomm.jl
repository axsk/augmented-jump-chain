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



function (m::NNModel)(x::AbstractMatrix)
    #crispboundary(x, m.nn(x), m.bndfun)
    #crispboundary2(x, m.nn(x), m.bndfun)
    softboundary(x, m.nn(x),  m.bndfun)
end

function softmodel(m, x)
    a = exp.(-30 * sum(abs2, x .- [ 1,0], dims=1)) # set a with bnd = 1
    b = exp.(-30 * sum(abs2, x .- [-1,0], dims=1)) # set b with bnd = 0
    r = 1 .- a .- b
    m.nn(x) .* r + a
end

function softboundary(x, y, _)
    a = exp.(-30 * sum(abs2, x .- [ 1,0], dims=1)) # set a with bnd = 1
    b = exp.(-30 * sum(abs2, x .- [-1,0], dims=1)) # set b with bnd = 0
    r = 1 .- a .- b
    y .* r + a
end

function crispboundary(x, y, bndfun)
    b = bndfun(x)
    y = map((b,y) -> isnan(b) ? y : b, b, y)
end

@adjoint function crispboundary(x, y, bndfun)
    dims = size(y)
    b = bndfun(x)
    y = map((b,y) -> isnan(b) ? y : b, b, y)
    y = reshape(y, dims)
    function pb(delta)
        delta = map((b,d) -> isnan(b) ? d : 0., b, delta)
        delta = reshape(delta, dims)
        (nothing, reshape(delta, size(y)), nothing)
    end
    y, pb
end


function crispboundary2(x, y, bndfun)
    b = bndfun(x)
    y = hook(g->replace(g, nothing=>0), y')
    map((b,y) -> isnan(b) ? y : b, b, y)
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

    #zs = hcat(([x,y] for x in xs, y in ys)...)
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


@adjoint triplewellbnd(x::AbstractMatrix) = triplewellbnd(x), d->(nothing, )#(zeros(size(@show d)),)

function train(iter=1000, batch=100;
    model = NNModel(
        Chain(Dense(2,10,sigmoid), Dense(10,10,sigmoid), Dense(10,1,sigmoid)),
        triplewellbnd),
    opt = ADAM(0.01),
    h = 0.1,
    plotevery=Inf)
    ps = Flux.params(model)
    losshist = []

    for i in 1:iter
        local losses
        x = rand(2,batch) .* [6, 4] .- [3,2]
        #xs = -3:.2:3
        #ys = -2:.2:2
        #x = hcat(([x,y] for x in xs, y in ys)...)

        l, pb = Flux.pullback(ps) do
            losses = sqraloss_batch(model, x, triplewell, h)
            s = sum(abs2,losses) / size(x, 2)
            #sum(sqraloss(model, x, triplewell, h) for x in eachcol(x)) / batch

        end
        #@show losses
        push!(losshist, l)
        println(l)
        grad = pb(1)
        Flux.Optimise.update!(opt, ps, grad)


        if i % plotevery == 0
            maxloss = max(maximum(losses))
            plot(model)# |> display
            p1 = Plots.scatter!(x[1,:], x[2,:], legend=false, marker=((losses' / maxloss).^(1/1) * 10, stroke(0)))
            p2 = Plots.plot(losshist, yscale=:log10)
            Plots.plot(p1, p2, layout=@layout([a;b]), size=(800,800)) |> display
        end
    end
    model
end

hyperwell(x) = 0.1*sum((x.^2 .-1).^2, dims=1)

struct Committorproblem{T}
    potential::T
    h::Float64
    beta::Float64
    a::Vector{Float64}
    b::Vector{Float64}
end

Committorproblem() = Committorproblem(triplewell, 0.1, 5., [1.,0], [-1.,0])

function hyperproblem(n=2; h=.1, beta=5.)
    a = ones(n)
    b = ones(n)
    b[1] = -1
    Committorproblem(hyperwell, h, beta, a, b)
end

defaultmodel(c::Committorproblem) = Chain(Dense(length(c.a), 10, σ), Dense(10, 10, σ), Dense(10, 1, σ))

function sample(c::Committorproblem, n)
    if length(c.a) == 2
        x = rand(2,n) .* [6, 4] .- [3,2]
    else
        x = rand(length(c.a),n) .* 4 .- 2
    end
end




import Zygote.@showgrad

function getboundary(c::Committorproblem, x)
    a = exp.(-30 * sum(abs2, x .- c.a, dims=1)) # set a with bnd = 1
    b = exp.(-30 * sum(abs2, x .- c.b, dims=1)) # set b with bnd = 0
    r = 1 .- a .- b
    a, b, r
end

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

function train2(batch=100, epochs=1000;
        c = Committorproblem(),
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
            p1 = Plots.scatter!(x[1,:], x[2,:], legend=false, marker=((pointlosses' / maxloss).^(1/1) * 10, stroke(0)))
            p2 = Plots.plot(losshist, yscale=:log10)
            Plots.plot(p1, p2, layout=@layout([a;b]), size=(800,800)) |> display
        end
    end
    model
end


end # module