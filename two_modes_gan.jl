using Distributions
using Plots
using Match
using JLD
include("optimizers_abstract.jl")
include("./GAN.jl")

n_hidden = 128
depth = 4
nonlin = ReLU
n_noise = 512
NUM_BATCH = 256
NUM_GRADS = 5000
ρ = 0.9
Z = zeros(n_noise, NUM_BATCH)
X = zeros(2, NUM_BATCH)

#Parameters to vary:
@show η = 0.025
#possible choices for algorithms are :GDA, :OGDA, :SGA, :conOpt, :CGDA
@show algo = :SGA

outfolder = "./out/twoMode_$(String(algo))_eta_$(Int(div(η, 0.001)))/"
# if isdir(outfolder)
#   rm(outfolder, recursive=true)
# end
mkdir(outfolder)


# Create mixture distribution
σ = 0.1
XDist = MixtureModel(MvNormal, [([1/√2, 1/√2], σ), ([1.0, 0.0], σ)])
ZDist = MvNormal(zeros(n_noise), 1.0)

G0 = init_denseNet(n_noise, 2, depth, n_hidden)
G = similar(G0)
ΔG = similar(G0)

D0 = init_denseNet(2, 1, depth, n_hidden)
D = similar(D0)
ΔD = similar(D0)

discriminator(weights, data) = denseNet(weights, data, 2, 1, depth, n_hidden, nonlin)
generator(weights, data) = denseNet(weights, data, n_noise, 2, depth, n_hidden, nonlin)

G_Loss(G, D) = - mean(crossEntTrue.(discriminator(D, X))) -
                 mean(crossEntFalse.(discriminator(D, generator(G, Z))))


D_Loss(G, D) =   mean(crossEntTrue.(discriminator(D, X))) +
                 mean(crossEntFalse.(discriminator(D, generator(G, Z))))


∇x_G, ∇y_G, ∇x∇x_G, ∇x∇y_G, ∇y∇y_G, ∇y∇x_G = get_closures(G_Loss)
∇x_D, ∇y_D, ∇x∇x_D, ∇x∇y_D, ∇y∇y_D, ∇y∇x_D = get_closures(D_Loss)


# Global parameters:
ε = 1e-6
γ = 1.0
ηG = η
ηD = η

newErrs = zeros(NUM_GRADS + 1)
newGrads = zeros(Int, NUM_GRADS + 1)

G .= G0
D .= D0
ΔG .= 0.0
ΔD .= 0.0
cumGrad = 0
op_noise = rand(ZDist, 1000)
ssg_G = ones(size(G))
ssg_D = ones(size(D))

for k = 1 : NUM_GRADS
  global ssg_G
  global ssg_D
  global cumGrad

  # Updating the randomness
  X .= rand(XDist, NUM_BATCH)
  Z .= rand(ZDist, NUM_BATCH)

  oldD = copy(D)
  gen_old = generator(G, Z)
  pb_old = discriminator(D, generator(G, Z))

  @match algo begin
    :CGDA => begin
      if iseven(k)
        newGrads[k+1] = CG_CGDA_x!(G, D, ΔG, ΔD,
                                    ∇x_G, ∇x∇y_G,
                                    ∇y_D, ∇y∇x_D, 
                                    ηG .* Diagonal(sqrt.(ssg_G.^(-1))), 
                                    ηD .* Diagonal(sqrt.(ssg_D.^(-1))), ε)
      else
        newGrads[k+1] = CG_CGDA_y!(G, D, ΔG, ΔD,
                                    ∇x_G, ∇x∇y_G,
                                    ∇y_D, ∇y∇x_D, 
                                    ηG .* Diagonal(sqrt.(ssg_G.^(-1))), 
                                    ηD .* Diagonal(sqrt.(ssg_D.^(-1))), ε)
      end
    end

    :OGDA => begin
      newGrads[k+1] = OGDA!(G, D, ΔG, ΔD, ∇x_G, ∇y_D, 
                            ηG .* Diagonal(sqrt.(ssg_G.^(-1))), 
                            ηD .* Diagonal(sqrt.(ssg_D.^(-1))))
    end

    :GDA => begin
      newGrads[k+1] = GDA!(G, D, ∇x_G, ∇y_D, 
                            ηG .* Diagonal(sqrt.(ssg_G.^(-1))), 
                            ηD .* Diagonal(sqrt.(ssg_D.^(-1))))
    end

    :conOpt => begin
      newGrads[k+1] = conOpt!(G, D, 
                              ∇x_G, ∇y_D, ∇x∇x_G, ∇x∇y_G, ∇y∇y_D, ∇y∇x_D,
                              ηG .* Diagonal(sqrt.(ssg_G.^(-1))), 
                              ηD .* Diagonal(sqrt.(ssg_D.^(-1))), γ)
    end

    :SGA => begin
      newGrads[k+1] = SGA!(G, D, 
                              ∇x_G, ∇y_D, ∇x∇y_G, ∇y∇x_D,
                              ηG .* Diagonal(sqrt.(ssg_G.^(-1))), 
                              ηD .* Diagonal(sqrt.(ssg_D.^(-1))), γ)
    end
  end



  #RMSPROP:
  ssg_G = ssg_G * ρ + (1.0 - ρ) * ∇x_G(G,D).^2
  ssg_D = ssg_D * ρ + (1.0 - ρ) * ∇y_D(G,D).^2

  gen_new = generator(G, Z)
  pb_new = discriminator(oldD, generator(G, Z))

  if mod(k-1, 1) == 0
    plt = scatter(vec(X[1,:]), vec(X[2,:]), xlim=(-1.5,1.5), ylim=(-1.5,1.5), 
    zcolor=vec(discriminator(oldD, X)), markershape=:utriangle,
    label = "True Data", title="Iteration $(k), $(cumGrad) Model Evaluations")
    scatter!(plt, vec(gen_old[1,:]), vec(gen_old[2,:]), zcolor=vec(pb_old),
             label="Fake Data")
    quiver!(plt, vec(gen_old[1,:]), vec(gen_old[2,:]), 
    quiver=(vec(gen_new[1,:]) - vec(gen_old[1,:]), 
            vec(gen_new[2,:]) - vec(gen_old[2,:])))
    savefig(plt, outfolder * "plot_iter_$(k)_grad_$(cumGrad)")
    @save  outfolder * "data_iter_$(k)_grad_$(cumGrad).jld" D G
  end

  @show cumGrad += newGrads[k+1]
  if cumGrad > NUM_GRADS
    resize!(newGrads, k)
    resize!(newErrs, k)
    break
  end
end
