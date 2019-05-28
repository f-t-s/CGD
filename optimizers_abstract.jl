using LinearAlgebra
using LinearMaps
using ReverseDiff
using ForwardDiff
using IterativeSolvers

# Gets all the necessary derivatives
function get_closures(f)

  function ∇x_f(x, y)
    return ReverseDiff.gradient((x, y) -> f(x, y), (x, y))[1]
  end

  function ∇y_f(x, y)
    return ReverseDiff.gradient((x, y) -> f(x, y), (x, y))[2]
  end

  function ∇y∇y_f(v, x, y)
    function ∇y_f(y, x)
      return ReverseDiff.gradient((y, x) -> f(x, y), (y, x))[1]
    end
    return ForwardDiff.derivative(h -> ∇y_f(y + h * v, x ), zero(eltype(v)))
  end

  function ∇y∇x_f(v, x, y)
    function ∇y_f(x, y) 
      return ReverseDiff.gradient((x, y) -> f(x, y), (x, y))[2]
    end
    return ForwardDiff.derivative(h -> ∇y_f(x + h * v, y ), zero(eltype(v)))
  end

  function ∇x∇x_f(v, x, y)
    function ∇x_f(x, y)
      return ReverseDiff.gradient((x, y) -> f(x, y), (x, y))[1]
    end
    return ForwardDiff.derivative(h -> ∇x_f(x + h * v, y ), zero(eltype(v)))
  end

  function ∇x∇y_f(v, x, y)
    function ∇x_f(y, x)
      return ReverseDiff.gradient((y, x) -> f(x, y), (y, x))[2]
    end
    return ForwardDiff.derivative(h -> ∇x_f(y + h * v, x ), zero(eltype(v)))
  end

  return ∇x_f, ∇y_f, ∇x∇x_f, ∇x∇y_f, ∇y∇y_f, ∇y∇x_f
end

# Gets all the necessary derivatives just using forward mode 
# autodiff. Slower for large problems, only used for debugging
function get_closures_forward(f)

  function ∇x_f(x, y)
    return ForwardDiff.gradient(x -> f(x, y), x)
  end

  function ∇y_f(x, y)
    return ForwardDiff.gradient(y -> f(x, y), y)
  end

  function ∇y∇y_f(v, x, y)
    function ∇y_f(y)
      return ForwardDiff.gradient(y -> f(x, y), y)
    end

    return ForwardDiff.derivative(h -> ∇y_f(y + h * v), zero(eltype(v)))
  end

  function ∇y∇x_f(v, x, y)
    function ∇y_f(x)
      return ForwardDiff.gradient(y -> f(x, y), y)
    end

    return ForwardDiff.derivative(h -> ∇y_f(x + h * v), zero(eltype(v)))
  end

  function ∇x∇x_f(v, x, y)
    function ∇x_f(x)
      return ForwardDiff.gradient(x -> f(x, y), x)
    end

    return ForwardDiff.derivative(h -> ∇x_f(x + h * v), zero(eltype(v)))
  end


  function ∇x∇y_f(v, x, y)
    function ∇x_f(y)
      return ForwardDiff.gradient(x -> f(x, y), x)
    end

    return ForwardDiff.derivative(h -> ∇x_f(y + h * v), zero(eltype(v)))
  end

  return ∇x_f, ∇y_f, ∇x∇x_f, ∇x∇y_f, ∇y∇y_f, ∇y∇x_f
end


function GDA!(x, y, ∇x_f, ∇y_g, η)
  ∇x_f = ∇x_f(x,y)
  ∇y_g = ∇y_g(x,y)
  x .-= η .* ∇x_f 
  y .-= η .* ∇y_g 
  return 2
end

function GDA!(x, y, ∇x_f, ∇y_g, ηx, ηy)
  ∇x_f = ∇x_f(x,y)
  ∇y_g = ∇y_g(x,y)
  x .-= ηx * ∇x_f 
  y .-= ηy * ∇y_g 
  return 2
end


function OGDA!(x, y, Δx, Δy, ∇x_f, ∇y_g, η )
  ∇x_f = ∇x_f(x,y)
  ∇y_g = ∇y_g(x,y)
  x .-= 2 * η * ∇x_f - η .* Δx
  y .-= 2 * η * ∇y_g - η .* Δy
  Δx .= ∇x_f
  Δy .= ∇y_g
  return 2
end

function OGDA!(x, y, Δx, Δy, ∇x_f, ∇y_g, ηx, ηy )
  ∇x_f = ∇x_f(x,y)
  ∇y_g = ∇y_g(x,y)
  x .-= 2 * ηx * ∇x_f - ηx * Δx
  y .-= 2 * ηy * ∇y_g - ηy * Δy
  Δx .= ∇x_f
  Δy .= ∇y_g
  return 2
end


#Only coincides with SGA in the case where f = -g
function SGA!(x, y, ∇x_f, ∇y_g, ∇x∇y_f, ∇y∇x_g, η, γ)
  ∇x_f = ∇x_f(x,y)
  ∇y_g = ∇y_g(x,y)
  Δx = ∇x_f - γ * ∇x∇y_f(∇y_g, x, y)
  Δy = ∇y_g - γ * ∇y∇x_g(∇x_f, x, y)
  x .-= η * Δx
  y .-= η * Δy
  return 4
end

function SGA!(x, y, ∇x_f, ∇y_g, ∇x∇y_f, ∇y∇x_g, ηx, ηy, γ)
  ∇x_f = ∇x_f(x,y)
  ∇y_g = ∇y_g(x,y)
  Δx = ∇x_f - γ * ∇x∇y_f(∇y_g, x, y)
  Δy = ∇y_g - γ * ∇y∇x_g(∇x_f, x, y)

  x .-= ηx * Δx
  y .-= ηy * Δy
  return 4
end

function conOpt!(x, y, ∇x_f, ∇y_g, ∇x∇x_f, ∇x∇y_f, ∇y∇y_g, ∇y∇x_g, η, γ)
  ∇x_f = ∇x_f(x,y)
  ∇y_g = ∇y_g(x,y)
  Δx = ∇x_f - γ * ∇x∇y_f(∇y_g, x, y) + γ * ∇x∇x_f(∇x_f, x, y)
  Δy = ∇y_g - γ * ∇y∇x_g(∇x_f, x, y) + γ * ∇y∇y_g(∇y_g, x, y)

  x .-= η * Δx
  y .-= η * Δy
  return 6
end

function conOpt!(x, y, ∇x_f, ∇y_g, ∇x∇x_f, ∇x∇y_f, ∇y∇y_g, ∇y∇x_g, ηx, ηy, γ)
  ∇x_f = ∇x_f(x,y)
  ∇y_g = ∇y_g(x,y)
  Δx = ∇x_f - γ * ∇x∇y_f(∇y_g, x, y) + γ * ∇x∇x_f(∇x_f, x, y)
  Δy = ∇y_g - γ * ∇y∇x_g(∇x_f, x, y) + γ * ∇y∇y_g(∇y_g, x, y)

  x .-= ηx * Δx
  y .-= ηy * Δy
  return 6
end


function LCGDA!(x, y, ∇x_f, ∇x∇y_f, ∇y_g, ∇y∇x_g, η)
  ∇x_f = ∇x_f(x, y)
  ∇y_g = ∇y_g(x, y)

  Δx = (∇x_f - η * ∇x∇y_f(∇y_g, x, y))
  Δy = (∇y_g - η * ∇y∇x_g(∇x_f, x, y))

  x .-= η * Δx
  y .-= η * Δy
  return 4
end


function CG_CGDA!(x, y, Δx, Δy, ∇x_f, ∇x∇y_f, ∇y_g, ∇y∇x_g, η, ε)
  ∇x_f = ∇x_f(x, y)
  ∇y_g = ∇y_g(x, y)

  #Defining the matvec for the solution of the system of player x
  function MVec_x(inp)
    yTemp = ∇y∇x_g(inp, x, y)
    return inp - η^2 * ∇x∇y_f(yTemp, x, y)
  end
  M_x = LinearMap{eltype(x)}(MVec_x, size(x,1), size(x,1))
  rhs_x = - (∇x_f - η * ∇x∇y_f(∇y_g, x, y))

  # Adjusting the tolerance of the algorithm, only necessary due to
  # limitations of package, hence not tracked.
  tolad_x = norm(rhs_x) / norm(MVec_x(Δx) - rhs_x)

  Δx .= rhs_x
  Δx, xhist = cg!(Δx, M_x, rhs_x; tol = tolad_x * ε, log = true)

  #Defining the matvec for the solution of the system of player y
  function MVec_y(inp)
    yTemp = ∇x∇y_f(inp, x, y)
    return inp - η^2 * ∇y∇x_g(yTemp, x, y)
  end
  M_y = LinearMap{eltype(x)}(MVec_y, size(y,1), size(y,1))
  rhs_y = - (∇y_g - η * ∇y∇x_g(∇x_f, x, y))

  # Adjusting the tolerance of the algorithm, only necessary due to
  # limitations of package, hence not tracked.
  tolad_y = norm(rhs_y) / norm(MVec_y(Δy) - rhs_y)

  Δy .= rhs_y
  Δy, yhist = cg!(Δy, M_y, rhs_y; tol = tolad_y * ε, log = true)
  
  x .+= η * Δx
  y .+= η * Δy
  return 2 * (xhist.iters  + yhist.iters) + 4
end

function CG_CGDA!(x, y, Δx, Δy, ∇x_f, ∇x∇y_f, ∇y_g, ∇y∇x_g, ηx, ηy, ε)
  ∇x_f = ∇x_f(x, y)
  ∇y_g = ∇y_g(x, y)

  #Defining the matvec for the solution of the system of player x
  function MVec_x(inp)
    yTemp = ηy * ∇y∇x_g(inp, x, y)
    return ηx \ inp - ∇x∇y_f(yTemp, x, y)
  end
  M_x = LinearMap{eltype(x)}(MVec_x, size(x,1), size(x,1))
  rhs_x = - (∇x_f - ∇x∇y_f(ηy * ∇y_g, x, y))

  # Adjusting the tolerance of the algorithm, only necessary due to
  # limitations of package, hence not tracked in computational complexity
  tolad_x = norm(rhs_x) / norm(MVec_x(Δx) - rhs_x)

  Δx .= rhs_x
  Δx, xhist = cg!(Δx, M_x, rhs_x; tol = tolad_x * ε, log = true)

  #Defining the matvec for the solution of the system of player y
  function MVec_y(inp)
    yTemp = ηx * ∇x∇y_f(inp, x, y)
    return ηy \ inp - ∇y∇x_g(yTemp, x, y)
  end
  M_y = LinearMap{eltype(x)}(MVec_y, size(y,1), size(y,1))
  rhs_y = - (∇y_g - ∇y∇x_g(ηx * ∇x_f, x, y))

  # Adjusting the tolerance of the algorithm, only necessary due to
  # limitations of package, hence not tracked.
  tolad_y = norm(rhs_y) / norm(MVec_y(Δy) - rhs_y)

  Δy .= rhs_y
  Δy, yhist = cg!(Δy, M_y, rhs_y; tol = tolad_y * ε, log = true)
  
  x .+= Δx
  y .+= Δy
  return 2 * (xhist.iters  + yhist.iters) + 4
end


#CGDA only computing the equilibrium for the first player
function CG_CGDA_x!(x, y, Δx, Δy, ∇x_f, ∇x∇y_f, ∇y_g, ∇y∇x_g, η, ε)
  ∇x_f = ∇x_f(x, y)
  ∇y_g = ∇y_g(x, y)

  #Defining the matvec for the solution of the system of player x
  function MVec_x(inp)
    yTemp = ∇y∇x_g(inp, x, y)
    return inp - η^2 * ∇x∇y_f(yTemp, x, y)
  end
  M_x = LinearMap{eltype(x)}(MVec_x, size(x,1), size(x,1))
  rhs_x = - (∇x_f - η * ∇x∇y_f(∇y_g, x, y))

  # Adjusting the tolerance of the algorithm, only necessary due to
  # limitations of package, hence not tracked.
  tolad_x = norm(rhs_x) / norm(MVec_x(Δx) - rhs_x)
  Δx, xhist = cg!(Δx, M_x, rhs_x; tol = tolad_x * ε, log = true)

  Δy .= - (∇y_g + η * ∇y∇x_g(Δx, x, y))
  
  x .+= η * Δx
  y .+= η * Δy
  return 2 * (xhist.iters ) + 4
end

function CG_CGDA_x!(x, y, Δx, Δy, ∇x_f, ∇x∇y_f, ∇y_g, ∇y∇x_g, ηx, ηy, ε)
  ∇x_f = ∇x_f(x, y)
  ∇y_g = ∇y_g(x, y)

  #Defining the matvec for the solution of the system of player x
  function MVec_x(inp)
    yTemp = ηy * ∇y∇x_g(inp, x, y)
    return ηx \ inp - ∇x∇y_f(yTemp, x, y)
  end
  M_x = LinearMap{eltype(x)}(MVec_x, size(x,1), size(x,1))
  rhs_x = - (∇x_f - ∇x∇y_f(ηy * ∇y_g, x, y))

  # Adjusting the tolerance of the algorithm, only necessary due to
  # limitations of package, hence not tracked.
  tolad_x = norm(rhs_x) / norm(MVec_x(Δx) - rhs_x)
  Δx, xhist = cg!(Δx, M_x, rhs_x; tol = tolad_x * ε, log = true)

  Δy .= - ηy * (∇y_g + ∇y∇x_g(Δx, x, y))
  
  x .+= Δx
  y .+= Δy
  return 2 * (xhist.iters ) + 4
end


#CGDA only computing the equilibrium for the second player
function CG_CGDA_y!(x, y, Δx, Δy, ∇x_f, ∇x∇y_f, ∇y_g, ∇y∇x_g, η, ε)
  ∇x_f = ∇x_f(x, y)
  ∇y_g = ∇y_g(x, y)


  #Defining the matvec for the solution of the system of player y
  function MVec_y(inp)
    yTemp = ∇x∇y_f(inp, x, y)
    return inp - η^2 * ∇y∇x_g(yTemp, x, y)
  end
  M_y = LinearMap{eltype(x)}(MVec_y, size(y,1), size(y,1))
  rhs_y = - (∇y_g - η * ∇y∇x_g(∇x_f, x, y))

  # Adjusting the tolerance of the algorithm, only necessary due to
  # limitations of package, hence not tracked.
  tolad_y = norm(rhs_y) / norm(MVec_y(Δy) - rhs_y)
  Δy, yhist = cg!(Δy, M_y, rhs_y; tol = tolad_y * ε, log = true)

  Δx .= - (∇x_f + η * ∇x∇y_f(Δy, x, y))
  
  x .+= η * Δx
  y .+= η * Δy
  return 2 * (yhist.iters) + 4
end

function CG_CGDA_y!(x, y, Δx, Δy, ∇x_f, ∇x∇y_f, ∇y_g, ∇y∇x_g, ηx, ηy, ε)
  ∇x_f = ∇x_f(x, y)
  ∇y_g = ∇y_g(x, y)


  #Defining the matvec for the solution of the system of player y
  function MVec_y(inp)
    yTemp = ηx * ∇x∇y_f(inp, x, y)
    return ηy \ inp - ∇y∇x_g(yTemp, x, y)
  end
  M_y = LinearMap{eltype(x)}(MVec_y, size(y,1), size(y,1))
  rhs_y = - (∇y_g - ∇y∇x_g(ηx * ∇x_f, x, y))

  # Adjusting the tolerance of the algorithm, only necessary due to
  # limitations of package, hence not tracked.
  tolad_y = norm(rhs_y) / norm(MVec_y(Δy) - rhs_y)
  Δy, yhist = cg!(Δy, M_y, rhs_y; tol = tolad_y * ε, log = true)

  Δx .= - ηx * (∇x_f + ∇x∇y_f(Δy, x, y))
  
  x .+= Δx
  y .+= Δy

  return 2 * (yhist.iters) + 4
end
