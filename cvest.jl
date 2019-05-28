# This file is used to run the experiments on the deterministic version of
# the covariance estimation problem.
include("./optimizers_abstract.jl")
using Distributions: MvNormal, mean
using Base. Iterators
using LinearAlgebra
using Plots
using Random 
using JLD

for (d,NUM_GRADS) in [(20,100000), 
                      (40,100000), 
                      (60, 100000)
                      ]
  # Global parameters:
  ε = 1e-6
  γ = 1.0
  ηList = [0.005, 0.025, 0.1, 0.4] 

  cdict = Dict(ηList[1] => :red, 
               ηList[2] => :green,
               ηList[3] => :blue,
               ηList[4] => :brown) 

  # Creating the problem: 
  U = randn(d,d) 
  Σ = U * U' 

  # Objective functions of the two players
  f(W,V) = - sum(reshape(W,d,d) .* Σ) + 
    sum(reshape(W,d,d) .* (reshape(V,d,d) * reshape(V,d,d)'))
  g(W,V) = + sum(reshape(W,d,d) .* Σ) -
    sum(reshape(W,d,d) .* (reshape(V,d,d) * reshape(V,d,d)'))

  # Get the derivatives
  ∇x_f, ∇y_f, ∇x∇x_f, ∇x∇y_f, ∇y∇y_f, ∇y∇x_f = get_closures(f)
  ∇x_g, ∇y_g, ∇x∇x_g, ∇x∇y_g, ∇y∇y_g, ∇y∇x_g = get_closures(g)

  # Function that returns the residual
  resi(W,V) = sqrt(norm(reshape(V,d,d) * reshape(V,d,d)' - Σ)^2 +
                   norm((reshape(W,d,d) + reshape(W,d,d)')/2)^2)

  # Creating the initial values as small perturbation of an equilibrium point
  W0 = rand(d,d) .- 0.5
  W0 = vec(W0 * W0')
  V0 = vec(rand(d,d) .- 0.5)
  W0 .= vec(zero(W0) + 1.0 * W0)
  V0 .= vec(U) + 1.0 * V0

  # Create the iteration variables
  V = similar(V0)
  ΔV = similar(V0)
  W = similar(W0)
  ΔW = similar(W0)
  
  # Start of the experiments:
  # Start an experiment: 

  errsCGDA = [] 
  gradsCGDA = [] 
  sgradsCGDA = [] 

  for η in ηList  
    newErrs = zeros(NUM_GRADS + 1)
    newGrads = zeros(Int, NUM_GRADS + 1)

    V .= V0
    W .= W0
    ΔW .= 0.0
    ΔV .= 0.0
    newErrs[1] = resi(W, V)
    cumGrad = 0
    for k = 1 : NUM_GRADS
      if iseven(k)
        newGrads[k+1] = CG_CGDA_x!(W, V, ΔW, ΔV,
                                    ∇x_f, ∇x∇y_f,
                                    ∇y_g, ∇y∇x_g, η, ε)
      else
        newGrads[k+1] = CG_CGDA_y!(W, V, ΔW, ΔV,
                                    ∇x_f, ∇x∇y_f,
                                    ∇y_g, ∇y∇x_g, η, ε)
      end
      cumGrad += newGrads[k+1]
      if cumGrad > NUM_GRADS
        resize!(newGrads, k)
        resize!(newErrs, k)
        break
      end
      newErrs[k + 1] = resi(W, V)
    end

    push!(errsCGDA, newErrs) 
    push!(gradsCGDA, newGrads) 
    push!(sgradsCGDA, cumsum(newGrads))
  end


  #Start an experiment: 

  errsOGDA = [] 
  gradsOGDA = [] 
  sgradsOGDA = [] 

  for η in ηList
    newErrs = zeros(NUM_GRADS + 1)
    newGrads = zeros(Int, NUM_GRADS + 1)

    V .= V0
    W .= W0
    ΔW .= 0.0
    ΔV .= 0.0
    newErrs[1] = resi(W, V)
    cumGrad = 0
    for k = 1 : NUM_GRADS
      newGrads[k+1] = OGDA!(W, V, ΔW, ΔV, ∇x_f, ∇y_g, η)
      newErrs[k + 1] = resi(W, V)
      cumGrad += newGrads[k+1]
      if cumGrad > NUM_GRADS
        resize!(newGrads, k)
        resize!(newErrs, k)
        break
      end

    end

    push!(errsOGDA, newErrs) 
    push!(gradsOGDA, newGrads) 
    push!(sgradsOGDA, cumsum(newGrads))
  end


  #Start an experiment: 
  errsSGA = [] 
  gradsSGA = [] 
  sgradsSGA = [] 

  for η in ηList
    newErrs = zeros(NUM_GRADS + 1)
    newGrads = zeros(Int, NUM_GRADS + 1)

    V .= V0
    W .= W0
    ΔW .= 0.0
    ΔV .= 0.0
    newErrs[1] = resi(W, V)
    cumGrad = 0
    for k = 1 : NUM_GRADS
      newGrads[k+1] = SGA!(W, V, ∇x_f, ∇y_g, ∇x∇y_f, ∇y∇x_g, η, γ)
      newErrs[k + 1] = resi(W, V)
      cumGrad += newGrads[k+1]
      if cumGrad > NUM_GRADS
        resize!(newGrads, k)
        resize!(newErrs, k)
        break
      end

    end

    push!(errsSGA, newErrs) 
    push!(gradsSGA, newGrads) 
    push!(sgradsSGA, cumsum(newGrads))
  end



  #Start an experiment: 
  errsConOpt = [] 
  gradsConOpt = [] 
  sgradsConOpt = [] 

  for η in ηList
    newErrs = zeros(NUM_GRADS + 1)
    newGrads = zeros(Int, NUM_GRADS + 1)


    V .= V0
    W .= W0
    ΔW .= 0.0
    ΔV .= 0.0
    newErrs[1] = resi(W, V)
    cumGrad = 0
    for k = 1 : NUM_GRADS
      newGrads[k+1] = conOpt!(W, V, ∇x_f, ∇y_g, 
                                  ∇x∇x_f, ∇x∇y_f,
                                  ∇y∇y_g, ∇y∇x_g, η, γ)
      newErrs[k + 1] = resi(W, V)
      cumGrad += newGrads[k+1]
      if cumGrad > NUM_GRADS
        resize!(newGrads, k)
        resize!(newErrs, k)
        break
      end

    end

    push!(errsConOpt, newErrs) 
    push!(gradsConOpt, newGrads) 
    push!(sgradsConOpt, cumsum(newGrads))
  end

  #Plotting the results: 
  cvgPlot = plot(xlim=(0:NUM_GRADS), 
                 xlabel="Number of forward passes",
                 ylabel="\$\\log_{10}(\\|V V^{T} - \\Sigma\\|_{Fro}^2 + \\|W_{SYM}\\|_{Fro}^2)/2\$",
                 legend=:bottomleft)

  for (sgrads, errs, η) in zip(sgradsCGDA, errsCGDA, ηList)
    mxg = findlast(x -> ~isnan(x), errs)
    if findfirst(isnan, errs) == nothing
      plot!(cvgPlot, sgrads[1:mxg], log10.(errs[1:mxg]),
            linestyle=:solid, linecolor=cdict[η], 
            label="CGD, \\eta = $η")
    end
  end

  for (sgrads, errs, η) in zip(sgradsOGDA, errsOGDA, ηList)
    mxg = findlast(x -> ~isnan(x), errs)
    if findfirst(isnan, errs) == nothing
      plot!(cvgPlot, sgrads[1:mxg], log10.(errs[1:mxg]),
            linestyle=:dot, linecolor=cdict[η],
            label="OGDA, \\eta = $η")
    end
  end

  for (sgrads, errs, η) in zip(sgradsSGA, errsSGA, ηList)
    mxg = findlast(x -> ~isnan(x), errs)
    if findfirst(isnan, errs) == nothing
      plot!(cvgPlot, sgrads[1:mxg], log10.(errs[1:mxg]),
            linestyle=:dash, linecolor=cdict[η],
            label="SGA, \\eta = $η")
    end
  end

  for (sgrads, errs, η) in zip(sgradsConOpt, errsConOpt, ηList)
    mxg = findlast(x -> ~isnan(x), errs)
    if findfirst(isnan, errs) == nothing
      plot!(cvgPlot, sgrads[1:mxg], log10.(errs[1:mxg]),
            linestyle=:dashdot, linecolor=cdict[η],
            label="ConOpt, \\eta = $η")
    end
  end

  savefig(cvgPlot, "./out/cvest_d_$(d)_maxIter_$NUM_GRADS") 
  @save "./out/cvest_d_$(d)_maxIter_$NUM_GRADS.jld" sgradsCGDA sgradsConOpt sgradsOGDA sgradsSGA gradsCGDA gradsConOpt gradsOGDA gradsSGA errsCGDA errsConOpt errsOGDA errsSGA ηList cdict 
end 


