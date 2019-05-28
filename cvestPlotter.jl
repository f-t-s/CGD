# Helper function to plot results for the 
# deterministic covariance estimation problem
using Plots
using JLD

#reset past font scaling
Plots.scalefontsizes()
#scale fonts
Plots.scalefontsizes(1.5)

@load "./out/cvest_d_60_maxIter_100000.jld"
NUM_GRADS = 100000
d = 40


cvgPlot = plot(xlim=(0:NUM_GRADS), 
                 xlabel="Number of forward passes",
                 ylabel="\$\\log_{10}(\\|V V^{T} - \\Sigma\\|_{Fro}^2 + \\|W W^{T}\\|_{Fro}^2)/2\$",
                 legend=:right)

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

  savefig(cvgPlot, "./out/cvest_d_$(d)") 
