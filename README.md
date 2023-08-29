# augmented-jump-chain (former name `birthdeath`)

Amongst the rest, most prominently features the code for the [The Augmented Jump Chain](https://doi.org/10.1002/adts.202000274)

## Contents
### Part 1 (python)
- **EAMC** + Paper Plots
- Hitting times + Optimization (ADAM, RProp, Momentum, RMSProp)
- Adjoint ODE solver as in SUNDIALS (for optimization of hitting times?)
- Temporal Gillespie
- SPA
- SQRA (ndtorus, perturbation, derivatives for the adjoint problem)

### Part 2 (julia) (most is now in Sqra.jl)
- ISOKANN experiments
- Committor neural network
- Autodiff Bug MWEs
- SparseBoxes, Sqra, picking
- voronoi neighborhood by linear program of H. Lie
- meta SGD
- adaptive euler maruyama

# History
WIP from 07.20 to 06.21

Continuation of https://github.com/axsk/generators and in a similar dirty state
Most usefull ideas where ported to ttps://github.com/axsk/Sqra.jl
