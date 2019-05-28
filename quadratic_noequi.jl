# Script to create the first three panels of Figure 3
 
using Plots 
include("./optimizers_abstract.jl")
#reset past font scaling
Plots.scalefontsizes()
#scale fonts
Plots.scalefontsizes(4.5)
using Plots.PlotMeasures

γ = 1.00
η = 0.20
ε = 1e-6

nIter = 100
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

α = 6
f(x,y) = -α * (x'*x - y'*y)
g(x,y) = + α * (x'*x - y'*y)

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

l10DistGDA = log10.(sqrt.(xHistGDA.^2 + yHistGDA.^2))[:]./2
l10DistOGDA = log10.(sqrt.(xHistOGDA.^2 + yHistOGDA.^2))[:]./2
l10DistCGDA = log10.(sqrt.(xHistCGDA.^2 + yHistCGDA.^2))[:]./2
l10DistLCGD = log10.(sqrt.(xHistLCGD.^2 + yHistLCGD.^2))[:]./2
l10DistConOpt = log10.(sqrt.(xHistConOpt.^2 + yHistConOpt.^2))[:]./2

pl = plot(# xlabel="Iterations",
          # ylabel="\$\\log_{10}(r)\$", 
          linewidth = 5.0,
          legend=:false,
          # bottom_margin= 10 * mm,
          # left_margin= 7 * mm,
          right_margin= 5 * mm,
          #top_margin= 15 * mm
          )

plot!(pl, l10DistGDA, label="GDA, SGA, LCGD, CGD", linewidth = 5.0)
plot!(pl, l10DistConOpt, label="ConOpt", linewidth = 5.0)
plot!(pl, l10DistOGDA, label="OGDA", linewidth = 5.0)

 savefig(pl, "./out/quadratic_noequi_alpha$α") 

