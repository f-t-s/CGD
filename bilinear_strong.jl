# This file is used to create the plots in Figure 2 of the main paper
using Plots 
#reset past font scaling
Plots.scalefontsizes()
#scale fonts
Plots.scalefontsizes(2.0)

include("./optimizers_abstract.jl")
ε = 1e-6
γ = 1.00
η = 0.20
xmin = -2.0
xmax = 2.0
ymin = -2.0
ymax = 2.0

nIter = 50
xHistGDA = zeros(1, nIter + 1)
yHistGDA = zeros(1, nIter + 1)
xHistLCGD = zeros(1, nIter + 1)
yHistLCGD = zeros(1, nIter + 1)
xHistConOpt = zeros(1, nIter + 1)
yHistConOpt = zeros(1, nIter + 1)
xHistOGDA = zeros(1, nIter + 1)
yHistOGDA = zeros(1, nIter + 1)
xHistCGDA = zeros(1, nIter + 1)
yHistCGDA = zeros(1, nIter + 1)

# α  in 1, 3, 6
α = 1
f(x,y) = α * x'*y
g(x,y) = - α * x'*y

x0 = [0.5]
y0 = [0.5]

xHistGDA[:,1] .= x0
yHistGDA[:,1] .= y0
xHistLCGD[:,1] .= x0
yHistLCGD[:,1] .= y0
xHistConOpt[:,1] .= x0
yHistConOpt[:,1] .= y0
xHistOGDA[:,1] .= x0
yHistOGDA[:,1] .= y0
xHistCGDA[:,1] .= x0
yHistCGDA[:,1] .= y0


∇x_f, ∇y_f, ∇x∇x_f, ∇x∇y_f, ∇y∇y_f, ∇y∇x_f = get_closures(f)
∇x_g, ∇y_g, ∇x∇x_g, ∇x∇y_g, ∇y∇y_g, ∇y∇x_g = get_closures(g)

ΔxC = zeros(1)
ΔyC = zeros(1)

ΔxO = zeros(1)
ΔyO = zeros(1)
for k = 1 : nIter
  global ΔxC
  global ΔyC
  global ΔxO
  global ΔyO
  x, y = xHistGDA[:, k], yHistGDA[:, k]
  GDA!(x,
       y,
       ∇x_f, ∇y_g, η)
  xHistGDA[:, k + 1], yHistGDA[:, k + 1] = x,y

  x, y = xHistOGDA[:, k], yHistOGDA[:, k]
  OGDA!(x,
        y,
        ΔxO, ΔyO, ∇x_f, ∇y_g, η)
  xHistOGDA[:, k + 1], yHistOGDA[:, k + 1] = x, y

  x, y = xHistCGDA[:, k], yHistCGDA[:, k]
  CG_CGDA!(x,
           y,
           ΔxC, ΔyC , ∇x_f, ∇x∇y_f, ∇y_g, ∇y∇x_g, η, ε)
  xHistCGDA[:, k + 1], yHistCGDA[:, k + 1] = x, y

  x, y = xHistLCGD[:, k], yHistLCGD[:, k]
  LCGDA!(x,
         y,
         ∇x_f, ∇x∇y_f, ∇y_g, ∇y∇x_g, η,)
  xHistLCGD[:, k + 1], yHistLCGD[:, k + 1] = x, y

  x, y = xHistConOpt[:, k], yHistConOpt[:, k] 
  conOpt!(x,
          y,
          ∇x_f, ∇y_g, ∇x∇x_f, ∇x∇y_f, ∇y∇y_g, ∇y∇x_g, η, γ)
  xHistConOpt[:, k + 1], yHistConOpt[:, k + 1] = x, y
end 

pl = plot(xlim=[xmin,xmax], ylim=[ymin,ymax], xlabel="\$x\$", ylabel="\$y\$",
          legend=:topleft)

plot!(pl, xHistGDA[1:1:end], yHistGDA[1:1:end], 
      marker=:cross, markersize=5.0, label="GDA", linestyle=:dot,
      linewidth=5.0)
plot!(pl, xHistLCGD[1:1:end], yHistLCGD[1:1:end], 
      marker=:ltriangle, markersize=5.0, label="LCGD", linestyle=:dash,
      linewidth=5.0)
plot!(pl, xHistConOpt[1:1:end], yHistConOpt[1:1:end], 
      marker=:star, markersize=5.0, label="SGA", linestyle=:dashdotdot,
      linewidth=5.0)
plot!(pl, xHistOGDA[1:1:end], yHistOGDA[1:1:end], 
      marker=:xcross, markersize=5.0, label="OGDA", linestyle=:dashdot,
      linewidth=5.0)
plot!(pl, xHistCGDA[1:1:end], yHistCGDA[1:1:end], 
      marker=:pentagon, markersize=5.0, label="CGD", linestyle=:solid,
      linewidth=5.0)

savefig(pl, "./out/bilinear_strong_alpha$α") 

